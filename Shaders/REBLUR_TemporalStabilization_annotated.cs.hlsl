/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR Temporal Stabilization（时域稳定化）
//
// 它位于 REBLUR 后段，处在：
//   TemporalAccumulation -> HistoryFix -> Blur -> PostBlur -> TemporalStabilization
//
// 这个 pass 不再负责“主历史重投影”的大决策；那些工作已经在
// `REBLUR_TemporalAccumulation.cs.hlsl` 中完成。这里的职责更像是：
//   1. 使用 `TemporalAccumulation` 传下来的 SMB / VMB footprint 信息，在上一帧的
//      stabilized luma history 上继续取样；
//   2. 结合当前帧局部 3x3 统计量（均值 / 方差）做 history clamp；
//   3. 通过 `ComputeAntilag` 与 `GetTemporalAccumulationParams` 控制稳定化强度；
//   4. 只对 **luma** 做稳定化，再把结果回写到 radiance / SH / DO 信号上。
//
// 需要特别强调：
//   - `gHistory_DiffLumaStabilized` / `gHistory_SpecLumaStabilized` 都是 `Texture2D<float>`，
//     保存的是“稳定化后的亮度历史”，不是完整 radiance。
//   - 真正的 anti-firefly 与 fast history clamping 主体在
//     `REBLUR_HistoryFix.cs.hlsl` 与 `REBLUR_TemporalAccumulation.cs.hlsl`；
//     本 pass 只做最后一层更保守的时域稳定化。
//   - 该 shader 会用于 RADIANCE / SH / DO 变体，但**不会用于纯 OCCLUSION 变体**；
//     这一点已通过 `Shaders.cfg` 和 `Source/Reblur.cpp` 交叉确认。
// =====================================================================================

// [注解] NRD 基础定义：导出宏、采样/打包、公共数学接口
#include "NRD.hlsli"

// [注解] 数学辅助库，`Common.hlsli` / `REBLUR_Common.hlsli` 的若干函数会依赖它
#include "ml.hlsli"

// [注解] REBLUR 全局配置：
//        包括 TS 使用的 CatRom 开关、antilag 模式、stabilization 常量、shared constants 等
#include "REBLUR_Config.hlsli"

// [注解] 当前 pass 的资源绑定声明。
//
// 关键点：
//   - `gIn_ViewZ` 在 Source 侧绑定的是 `Permanent::PREV_VIEWZ`，
//     但这张纹理是在 `REBLUR_Blur.cs.hlsl` 中由当前帧 `IN_VIEWZ` 拷贝出来的，
//     所以它在这里的语义更接近“供后段和下帧复用的 depth copy”，不是“真正旧一帧的 viewZ”。
//   - `gInOut_Mv` 以 UAV 方式绑定到 `ResourceType::IN_MV`，但本 shader 里只读不写；
//     它本质上是一个 in-place / inout 资源绑定习惯，而不是这里主动生成 motion vectors。
#include "REBLUR_TemporalStabilization.resources.hlsli"

// [注解] 通用几何与采样工具：
//        `GetSpecMagicCurve`、`GetXvirtual`、几何重建等都在这里定义
#include "Common.hlsli"

// [注解] REBLUR 专属工具：
//        `UnpackData1/2`、`PackInternalData`、`ComputeAntilag`、
//        `GetTemporalAccumulationParams`、带 custom weights 的 history filter 等均在这里
#include "REBLUR_Common.hlsli"

// =====================================================================================
// [注解] shared memory：只缓存当前帧的局部 luma
//
// TemporalStabilization 的局部统计只依赖 3x3 邻域亮度，因此只需要缓存：
//   - `s_DiffLuma`
//   - `s_SpecLuma`
//
// 如果某个邻居像素超出 denoising range，则写入 `REBLUR_INVALID`，
// 后续在统计 moments 时会回退到中心像素的 luma，避免 INF/背景像素破坏方差估计。
// =====================================================================================
groupshared float s_DiffLuma[ BUFFER_Y ][ BUFFER_X ];
groupshared float s_SpecLuma[ BUFFER_Y ][ BUFFER_X ];

// =====================================================================================
// [注解] Preload：把当前帧 luma 预读进 shared memory
//
// 这里读的是 `gIn_Diff` / `gIn_Spec`（即 PostBlur 输出、TS 输入的“当前帧结果”），
// 不是历史缓冲。因为本 pass 的局部统计量要反映**当前帧空间邻域**的亮度分布。
// =====================================================================================
void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSizeMinusOne );

    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( globalPos ) ] );

    #if( NRD_DIFF )
        float diffLuma = GetLuma( gIn_Diff[ globalPos ] );
        s_DiffLuma[ sharedPos.y ][ sharedPos.x ] = viewZ > gDenoisingRange ? REBLUR_INVALID : diffLuma;
    #endif

    #if( NRD_SPEC )
        float specLuma = GetLuma( gIn_Spec[ globalPos ] );
        s_SpecLuma[ sharedPos.y ][ sharedPos.x ] = viewZ > gDenoisingRange ? REBLUR_INVALID : specLuma;
    #endif
}

// =====================================================================================
// [注解] 主 Compute Shader 入口
//
// 线程组尺寸来自 `REBLUR_TemporalStabilization.resources.hlsli`：
//   GROUP_X = 8
//   GROUP_Y = 16
//
// 与 TemporalAccumulation 不同，这里使用 `NRD_CTA_ORDER_REVERSED`；
// 这是 NRD 内部针对波前/缓存局部性的一个 CTA 排列策略，不改变算法语义。
// =====================================================================================
[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_REVERSED;

    // Preload
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // [注解] Tile-based early out：天空 tile 或越界线程直接退出
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // [注解] 逐像素 early out：超出 denoising range 的像素完全不写输出。
    //        源码特别注明了：这些像素必须依赖后续/外层通过 `viewZ` 自行拒绝，
    //        不能假设 UAV 被写成 0。
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pixelPos ) ] );
    if( viewZ > gDenoisingRange )
        return; // IMPORTANT: no data output, must be rejected by the "viewZ" check!

    // =====================================================================================
    // [注解] 当前像素位置重建
    //
    // `Xv`：当前像素的 view-space 位置
    // `X` ：当前像素的 world-space 位置
    //
    // 后面 specular 的 virtual motion 仍然会用到完整几何位置，而 diffuse 只需要 surface motion。
    // =====================================================================================
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );
    float3 X = Geometry::RotateVector( gViewToWorld, Xv );

    // =====================================================================================
    // [注解] 使用 motion vectors 估计上一帧表面位置（SMB 坐标）
    //
    // 注意：
    //   - `gInOut_Mv` 在资源声明里是 RWTexture，但这里实际只读取；
    //   - `mv.xy` 给出屏幕空间上的 surface motion 偏移；
    //   - `mv.z` 可能直接提供，也可能需要由 `gWorldToViewPrev` 反推；
    //   - 如果 motion vector 是 world-space 编码，则直接把 `mv` 加到 `X`。
    //
    // 最终得到：
    //   - `Xprev`：当前表面点在上一帧的世界空间估计
    //   - `smbPixelUv`：该点投到上一帧屏幕上的 uv
    // =====================================================================================
    float4 inMv = gInOut_Mv[ WithRectOrigin( pixelPos ) ];
    float3 mv = inMv.xyz * gMvScale.xyz;
    float3 Xprev = X;
    float2 smbPixelUv = pixelUv + mv.xy;

    if( gMvScale.w == 0.0 )
    {
        if( gMvScale.z == 0.0 )
            mv.z = Geometry::AffineTransform( gWorldToViewPrev, X ).z - viewZ;

        float viewZprev = viewZ + mv.z;
        float3 Xvprevlocal = Geometry::ReconstructViewPosition( smbPixelUv, gFrustumPrev, viewZprev, gOrthoMode ); // TODO: use gOrthoModePrev

        Xprev = Geometry::RotateVectorInverse( gWorldToViewPrev, Xvprevlocal ) + gCameraDelta.xyz;
    }
    else
    {
        Xprev += mv;
        smbPixelUv = Geometry::GetScreenUv( gWorldToClipPrev, Xprev );
    }

    // [注解] 当前像素的法线、粗糙度、材质 ID
    //        specular 分支的 virtual motion 与 responsive acceleration 都会用到它们。
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ], materialID );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    // =====================================================================================
    // [注解] 解包 `TemporalAccumulation` 传下来的中间状态
    //
    // `data1`（见 `PackData1/UnpackData1`）：
    //   - `data1.x` = diffuse accumulation speed（当前帧阶段）
    //   - `data1.y` = specular accumulation speed（当前帧阶段）
    //
    // `data2`（见 `PackData2/UnpackData2`）：
    //   - `bits[0..3]`   = SMB footprint 的 2x2 occlusion bitmask
    //   - `bits[4..7]`   = VMB footprint 的 2x2 occlusion bitmask（仅 spec）
    //   - `data2.x`      = `virtualHistoryAmount`
    //   - `data2.y`      = `curvature`
    //   - 另有 `smbAllowCatRom` bit
    //
    // 因为 TS 只消费这些状态，不再重新做大规模 reprojection 判定，所以这里必须严格按 TA 的打包语义解释。
    // =====================================================================================
    uint bits;
    bool smbAllowCatRom;
    REBLUR_DATA1_TYPE data1 = UnpackData1( gIn_Data1[ pixelPos ] );
    float2 data2 = UnpackData2( gIn_Data2[ pixelPos ], bits, smbAllowCatRom );

    // =====================================================================================
    // [注解] Surface motion footprint
    //
    // `smbOcclusion` 是 TA 已经写好的 2x2 可用性 bitmask；
    // 这里不会重新判断“哪些 tap 可用”，只负责把这些 bit 转成采样权重。
    //
    // `smbFootprintQuality` 是该 2x2 footprint 的整体完整度：
    //   - 1 表示四个 tap 都可信
    //   - 越小表示 footprint 越残缺，后续 antilag / temporal params 会更保守
    // =====================================================================================
    Filtering::Bilinear smbBilinearFilter = Filtering::GetBilinearFilter( smbPixelUv, gRectSizePrev );
    float4 smbOcclusion = float4( ( bits & uint4( 1, 2, 4, 8 ) ) != 0 );

    float4 smbOcclusionWeights = Filtering::GetBilinearCustomWeights( smbBilinearFilter, smbOcclusion );
    float smbFootprintQuality = Filtering::ApplyBilinearFilter( smbOcclusion.x, smbOcclusion.y, smbOcclusion.z, smbOcclusion.w, smbBilinearFilter );
    smbFootprintQuality = Math::Sqrt01( smbFootprintQuality );

    int2 smemPos = threadPos + NRD_BORDER;

    // =====================================================================================
    // [注解] ===================== Diffuse 稳定化路径 =====================
    //
    // Diffuse 没有 VMB，只沿 surface motion 采 stabilized luma history。
    // 算法流程：
    //   1. 在当前帧 3x3 邻域上统计 luma 的均值 / 方差；
    //   2. 若 HistoryFix 刚介入过，则先对当前 luma 做一层 firefly 清理；
    //   3. 沿 SMB 取上一帧 stabilized luma history；
    //   4. 做 antilag 与 history clamp；
    //   5. 只混合 luma，再回写到完整 `REBLUR_TYPE` / `REBLUR_SH_TYPE`。
    // =====================================================================================
    #if( NRD_DIFF )
        float diffLuma = s_DiffLuma[ smemPos.y ][ smemPos.x ];
        float diffLumaM1 = diffLuma;
        float diffLumaM2 = diffLuma * diffLuma;

        [unroll]
        for( j = 0; j <= NRD_BORDER * 2; j++ )
        {
            [unroll]
            for( i = 0; i <= NRD_BORDER * 2; i++ )
            {
                if( i == NRD_BORDER && j == NRD_BORDER )
                    continue;

                int2 pos = threadPos + int2( i, j );

                // Accumulate moments
                float d = s_DiffLuma[ pos.y ][ pos.x ];
                d = d == REBLUR_INVALID ? diffLuma : d;

                diffLumaM1 += d;
                diffLumaM2 += d * d;
            }
        }

        // [注解] 3x3 邻域的均值与标准差，用于后面的 antilag / history clamp
        diffLumaM1 /= ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 );
        diffLumaM2 /= ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 );

        float diffLumaSigma = GetStdDev( diffLumaM1, diffLumaM2 );

        // [注解] 如果 `data1.x < gHistoryFixFrameNum`，说明 HistoryFix 近期仍在介入。
        //        这里会对当前 luma 做一个温和的上限收缩，避免高亮离群值继续污染 stabilized history。
        if( data1.x < gHistoryFixFrameNum )
            diffLuma = min( diffLuma, diffLumaM1 * ( 1.2 + 1.0 / ( 1.0 + data1.x ) ) );

        // [注解] 采样上一帧 stabilized luma history。
        //        这里只采 `float` 历史，而不是整份 radiance。
        float diffLumaHistory;

        BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights1(
            saturate( smbPixelUv ) * gRectSizePrev, gResourceSizeInvPrev,
            smbOcclusionWeights, smbAllowCatRom,
            gHistory_DiffLumaStabilized, diffLumaHistory
        );

        // Avoid negative values
        diffLumaHistory = max( diffLumaHistory, 0.0 );

        // [注解] antilag 用“历史值 vs 当前局部均值”的差异来决定是否缩短历史。
        //        `ComputeAntilag` 实现在 `REBLUR_Common.hlsli`。
        float diffAntilag = ComputeAntilag( diffLumaHistory, diffLumaM1, diffLumaSigma, smbFootprintQuality * data1.x );

        float diffMinAccumSpeed = min( data1.x, gHistoryFixFrameNum ) * REBLUR_USE_ANTILAG_NOT_INVOKING_HISTORY_FIX;
        data1.x = lerp( diffMinAccumSpeed, data1.x, diffAntilag );

        // [注解] `GetTemporalAccumulationParams` 返回：
        //   x = 历史权重
        //   y = sigma 扩张因子（决定 history clamp 盒子的宽度）
        float2 diffTemporalAccumulationParams = GetTemporalAccumulationParams( smbFootprintQuality, data1.x, diffAntilag );

        float diffHistoryWeight = diffTemporalAccumulationParams.x;
        diffHistoryWeight *= float( pixelUv.x >= gSplitScreen );
        diffHistoryWeight *= float( smbPixelUv.x >= gSplitScreenPrev );

        // [注解] history clamp：把取回来的 stabilized history 限制在“当前 3x3 局部统计盒”里。
        diffLumaHistory = Color::Clamp( diffLumaM1, diffLumaSigma * diffTemporalAccumulationParams.y, diffLumaHistory );

        // [注解] `gStabilizationStrength` 来自 `maxStabilizedFrameNum / (1 + maxStabilizedFrameNum)`，
        //        如果 history reset，则 C++ 侧会把它直接置 0。
        float diffLumaStabilized = lerp( diffLuma, diffLumaHistory, min( diffHistoryWeight, gStabilizationStrength ) );

        // [注解] 只替换 luma，不直接改颜色方向；`ChangeLuma` 会按当前模式处理：
        //        - RADIANCE：改颜色亮度
        //        - DO：`.w` 就是“亮度/距离语义”，这里实际改的是 hit distance 通道
        REBLUR_TYPE diff = gIn_Diff[ pixelPos ];
        diff = ChangeLuma( diff, diffLumaStabilized );
        #if( NRD_MODE == SH )
            // [注解] SH 模式并不是独立稳定一套 SH 系数；
            //        而是先稳定主 luma，再按能量比例缩放 SH。
            REBLUR_SH_TYPE diffSh = gIn_DiffSh[ pixelPos ];
            diffSh *= GetLumaScale( length( diffSh ), diffLumaStabilized );
        #endif

        // [注解] 调试模式可把 `.w` 改写为 history length，而不是原本的 occlusion / hit distance。
        diff.w = gReturnHistoryLengthInsteadOfOcclusion ? data1.x : diff.w;

        gOut_Diff[ pixelPos ] = diff;
        gOut_DiffLumaStabilized[ pixelPos ] = diffLumaStabilized;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ pixelPos ] = diffSh;
        #endif
    #endif

    // =====================================================================================
    // [注解] ===================== Specular 稳定化路径 =====================
    //
    // Specular 比 diffuse 多了一层 virtual motion：
    //   - SMB：沿表面运动取历史
    //   - VMB：沿虚拟反射运动取历史
    //
    // 但在 TS 阶段，SMB / VMB 的判定结果已经由 `TemporalAccumulation` 写进 `data2`。
    // 这里主要做的是：
    //   1. 解包 `virtualHistoryAmount`、`curvature` 与 VMB occlusion bits；
    //   2. 用 `GetXvirtual` 重新构造虚拟点，得到 `vmbPixelUv`；
    //   3. 在 SMB/VMB 两条 stabilized luma history 采样路径之间选择；
    //   4. 再做 antilag、history clamp 和响应性加速。
    // =====================================================================================
    #if( NRD_SPEC )
        float specLuma = s_SpecLuma[ smemPos.y ][ smemPos.x ];
        float specLumaM1 = specLuma;
        float specLumaM2 = specLuma * specLuma;

        [unroll]
        for( j = 0; j <= NRD_BORDER * 2; j++ )
        {
            [unroll]
            for( i = 0; i <= NRD_BORDER * 2; i++ )
            {
                if( i == NRD_BORDER && j == NRD_BORDER )
                    continue;

                int2 pos = threadPos + int2( i, j );

                // Accumulate moments
                float s = s_SpecLuma[ pos.y ][ pos.x ];
                s = s == REBLUR_INVALID ? specLuma : s;

                specLumaM1 += s;
                specLumaM2 += s * s;
            }
        }

        // [注解] 和 diffuse 同理，先得到当前帧 3x3 邻域的局部统计量
        specLumaM1 /= ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 );
        specLumaM2 /= ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 );

        float specLumaSigma = GetStdDev( specLumaM1, specLumaM2 );

        // [注解] HistoryFix 近期介入过时，同样先压制当前帧的高亮离群值
        if( data1.y < gHistoryFixFrameNum )
            specLuma = min( specLuma, specLumaM1 * ( 1.2 + 1.0 / ( 1.0 + data1.y ) ) );

        // [注解] `hitDistForTracking` 由 TA / PrePass 维护。
        //        这里再读取它，是为了按当前的 `curvature` 重新构造 `Xvirtual`。
        //        也就是说，TS 不重新估计 tracking distance，只复用已有 tracking 语义。
        float hitDistForTracking = gIn_SpecHitDistForTracking[ pixelPos ];

        // [注解] `virtualHistoryAmount` 与 `curvature` 来自 `UnpackData2`：
        //        - `virtualHistoryAmount` 在正常路径下是 TA 决定的 SMB/VMB 选择结果（通常为 0 或 1）
        //        - `curvature` 是 TA 沿运动方向估计的局部表面曲率
        float virtualHistoryAmount = data2.x;
        float curvature = data2.y;

        float3 V = GetViewVector( X );
        float NoV = abs( dot( N, V ) );
        float3 Xvirtual = GetXvirtual( hitDistForTracking, curvature, X, Xprev, N, V, roughness );

        float2 vmbPixelUv = Geometry::GetScreenUv( gWorldToClipPrev, Xvirtual );
        vmbPixelUv = materialID == gCameraAttachedReflectionMaterialID ? pixelUv : vmbPixelUv;

        // [注解] VMB 的 2x2 occlusion bitmask 同样是 TA 预先写好的；
        //        TS 这里只把 bit 展开成采样权重，不重新做 plane/material 测试。
        Filtering::Bilinear vmbBilinearFilter = Filtering::GetBilinearFilter( vmbPixelUv, gRectSizePrev );
        float4 vmbOcclusion = float4( ( bits & uint4( 16, 32, 64, 128 ) ) != 0 );
        float4 vmbOcclusionWeights = Filtering::GetBilinearCustomWeights( vmbBilinearFilter, vmbOcclusion );

        // [注解] 只有当 footprint 足够完整时才允许 CatRom；
        //        并且 VMB 还要求 SMB 也允许 CatRom，以减少 disocclusion 区域过锐造成的错觉。
        bool vmbAllowCatRom = dot( vmbOcclusion, 1.0 ) > 3.5 && REBLUR_USE_CATROM_FOR_VIRTUAL_MOTION_IN_TS;
        vmbAllowCatRom = vmbAllowCatRom && smbAllowCatRom; // helps to reduce over-sharpening in disoccluded areas

        float vmbFootprintQuality = Filtering::ApplyBilinearFilter( vmbOcclusion.x, vmbOcclusion.y, vmbOcclusion.z, vmbOcclusion.w, vmbBilinearFilter );
        vmbFootprintQuality = Math::Sqrt01( vmbFootprintQuality );

        // [注解] `virtualHistoryAmount` 决定到底沿 SMB 还是 VMB 去采 stabilized luma history。
        //        如果该值为 0，就是纯 SMB；为 1，则纯 VMB。
        float2 uv = lerp( smbPixelUv, vmbPixelUv, virtualHistoryAmount );
        float4 occlusionWeights = lerp( smbOcclusionWeights, vmbOcclusionWeights, virtualHistoryAmount );
        bool allowCatRom = virtualHistoryAmount < 0.5 ? smbAllowCatRom : vmbAllowCatRom;

        float specLumaHistory;

        BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights1(
            saturate( uv ) * gRectSizePrev, gResourceSizeInvPrev,
            occlusionWeights, allowCatRom,
            gHistory_SpecLumaStabilized, specLumaHistory
        );

        // Avoid negative values
        specLumaHistory = max( specLumaHistory, 0.0 );

        // [注解] specular 的 antilag 使用 SMB/VMB 混合后的 footprint quality；
        //        VMB 越残缺、history 越不可信，antilag 越容易拉低有效 history length。
        float footprintQuality = lerp( smbFootprintQuality, vmbFootprintQuality, virtualHistoryAmount );
        float specAntilag = ComputeAntilag( specLumaHistory, specLumaM1, specLumaSigma, footprintQuality * data1.y );

        float specMinAccumSpeed = min( data1.y, gHistoryFixFrameNum ) * REBLUR_USE_ANTILAG_NOT_INVOKING_HISTORY_FIX;
        data1.y = lerp( specMinAccumSpeed, data1.y, specAntilag );

        float2 specTemporalAccumulationParams = GetTemporalAccumulationParams( footprintQuality, data1.y, specAntilag );

        // TODO: roughness should affect stabilization:
        // - use "virtualHistoryRoughnessBasedConfidence" from TA
        // - compute moments for samples with similar roughness
        float specHistoryWeight = specTemporalAccumulationParams.x;
        specHistoryWeight *= float( pixelUv.x >= gSplitScreen );
        specHistoryWeight *= virtualHistoryAmount != 1.0 ? float( smbPixelUv.x >= gSplitScreenPrev ) : 1.0;
        specHistoryWeight *= virtualHistoryAmount != 0.0 ? float( vmbPixelUv.x >= gSplitScreenPrev ) : 1.0;

        // [注解] 响应性加速：
        //        specular 越光滑（roughness 越低），通常越不能让 stabilized history 过重，
        //        否则会拖出明显高光拖影；
        //        `GetSpecMagicCurve` 与 `RemapRoughnessToResponsiveFactor` 正是在调这个权衡。
        float responsiveFactor = RemapRoughnessToResponsiveFactor( roughness );
        float smc = GetSpecMagicCurve( roughness );
        float acceleration = lerp( smc, 1.0, 0.5 + responsiveFactor * 0.5 );
        if( materialID == gStrandMaterialID )
            acceleration = min( acceleration, 0.5 );

        specHistoryWeight *= acceleration;

        // [注解] 同样只在当前局部统计盒内保留 stabilized history。
        specLumaHistory = Color::Clamp( specLumaM1, specLumaSigma * specTemporalAccumulationParams.y, specLumaHistory );

        float specLumaStabilized = lerp( specLuma, specLumaHistory, min( specHistoryWeight, gStabilizationStrength ) );

        REBLUR_TYPE spec = gIn_Spec[ pixelPos ];
        spec = ChangeLuma( spec, specLumaStabilized );
        #if( NRD_MODE == SH )
            REBLUR_SH_TYPE specSh = gIn_SpecSh[ pixelPos ];
            specSh *= GetLumaScale( length( specSh ), specLumaStabilized );
        #endif

        // [注解] 同 diffuse，调试模式下可把 `.w` 替换成当前 history length
        spec.w = gReturnHistoryLengthInsteadOfOcclusion ? data1.y : spec.w;

        gOut_Spec[ pixelPos ] = spec;
        gOut_SpecLumaStabilized[ pixelPos ] = specLumaStabilized;
        #if( NRD_MODE == SH )
            gOut_SpecSh[ pixelPos ] = specSh;
        #endif
    #endif

    // =====================================================================================
    // [注解] 输出 `InternalData`
    //
    // `PackInternalData` 在 `REBLUR_Common.hlsli` 中定义，语义非常关键：
    //   - 它不是把 `data1.x / data1.y` 原样打包；
    //   - 而是先把 diff/spec accumulation speed **各自 +1 并 clamp**，
    //     再和 material ID 一起压缩进 `uint`。
    //
    // 因此这里写出的 `gOut_InternalData` 本质上是“供下一帧使用的历史长度状态”。
    // 这也是为什么它会在 Source 侧被命名为 `PREV_INTERNAL_DATA`。
    // =====================================================================================
    gOut_InternalData[ pixelPos ] = PackInternalData( data1.x, data1.y, materialID );
}
