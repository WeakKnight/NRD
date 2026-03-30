/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR History Fix（历史修复）
//
// 它位于 REBLUR 管线的中后段：
//   TemporalAccumulation -> HistoryFix -> Blur -> PostBlur -> TemporalStabilization
//
// 这个 pass **不负责重新做完整的时域重投影**，那部分工作已经在
// `REBLUR_TemporalAccumulation.cs.hlsl` 中完成。
//
// HistoryFix 的职责更像是：
//   1. **修补历史还不够长的像素**
//      - 刚刚出现、刚 disocclusion、或者刚切换历史路径的像素，时域累计帧数比较短
//      - 此时主 history 仍然不稳定，容易有 blotchy / 局部不连续
//      - HistoryFix 会按一个“稀疏 stride 邻域”重新聚合周围更稳定的样本来补洞
//
//   2. **维护 fast history 并做亮度方差夹紧**
//      - `gIn_DiffFast` / `gIn_SpecFast` 保存的是更短、更敏捷的亮度历史
//      - 本 pass 会把修复后的 normal history 与 fast history 结合
//      - 再根据局部方差做 clamping，抑制 crawling、亮斑、局部爆闪
//
//   3. **可选 anti-firefly**
//      - 在更大的邻域上估计均值 / 方差
//      - 把当前 luma 限制在统计区间中，抑制稀疏高亮孤点
//
// 直观理解：
//   - TemporalAccumulation 决定“历史从哪来、混多少”
//   - HistoryFix 负责“历史太短时，怎么用邻域更稳妥地补起来”
//   - 后面的 Blur / PostBlur 再继续做更常规的空间整理
//
// 这个 pass 和 `REBLUR_TemporalStabilization.cs.hlsl` 很像，都会：
//   - preload 一份局部 luma 到 shared memory
//   - 做局部方差估计
//   - 只对 luma 做 clamp / 修正
// 但两者的目标不同：
//   - HistoryFix 偏向“短历史修补 + fast history 校正”
//   - TemporalStabilization 偏向“最终时域稳定化”
// =====================================================================================

// [注解] NRD 基础定义：导出宏、纹理访问宏、常用类型
#include "NRD.hlsli"

// [注解] 数学库：近似 acos、sqrt、smoothstep、pow 等，后面会频繁用到
#include "ml.hlsli"


// [注解] REBLUR 全局配置：
//        包括 HistoryFix / FastHistory / AntiFirefly 相关的关键宏和常量：
//          - REBLUR_HISTORY_FIX_FILTER_RADIUS
//          - REBLUR_FAST_HISTORY_CLAMPING_RADIUS
//          - REBLUR_ANTI_FIREFLY_FILTER_RADIUS
//          - REBLUR_ANTI_FIREFLY_SIGMA_SCALE
//          - gHistoryFixFrameNum / gHistoryFixBasePixelStride / gFastHistoryClampingSigmaScale
#include "REBLUR_Config.hlsli"

// [注解] 当前 pass 的资源绑定：
//        输入：tiles / normal+roughness / data1 / viewZ / diff/spec / diffFast/specFast / specHitDistForTracking / SH
//        输出：修复后的 diff/spec，以及更新后的 diffFast/specFast
#include "REBLUR_HistoryFix.resources.hlsli"

// [注解] 通用几何与采样工具：位置重建、UV 镜像、法线旋转、luma 操作等
#include "Common.hlsli"

// [注解] REBLUR 公共工具：
//        Pack/UnpackData1、GetSpecMagicCurve、hit distance normalization、权重函数等
#include "REBLUR_Common.hlsli"

// =====================================================================================
// [注解] shared memory：只缓存 fast history 的 luma
//
// 这里没有缓存完整 radiance，而只缓存：
//   - `s_DiffLuma`
//   - `s_SpecLuma`
//
// 原因：
//   HistoryFix 后半段需要对 fast history 做局部均值 / 方差统计，
//   统计对象只是亮度，不需要完整颜色。
//
// 如果某个 preload 像素超出降噪范围，则写入 `REBLUR_INVALID`，
// 后续在统计 moments 时会回退到中心值，避免背景/远距离数据污染方差估计。
// =====================================================================================
groupshared float s_DiffLuma[ BUFFER_Y ][ BUFFER_X ];
groupshared float s_SpecLuma[ BUFFER_Y ][ BUFFER_X ];

// =====================================================================================
// [注解] Preload：把邻域 fast history luma 装进 shared memory
//
// 注意这里缓存的是：
//   - `gIn_DiffFast`
//   - `gIn_SpecFast`
//
// 不是主 history，也不是当前 radiance。因为本 pass 的局部 clamp 逻辑依赖的是
// “快速响应的亮度历史分布”。
// =====================================================================================
void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSizeMinusOne );

    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( globalPos ) ] );

    #if( NRD_DIFF )
        float diffFast = gIn_DiffFast[ globalPos ];
        s_DiffLuma[ sharedPos.y ][ sharedPos.x ] = viewZ > gDenoisingRange ? REBLUR_INVALID : diffFast;
    #endif

    #if( NRD_SPEC )
        float specFast = gIn_SpecFast[ globalPos ];
        s_SpecLuma[ sharedPos.y ][ sharedPos.x ] = viewZ > gDenoisingRange ? REBLUR_INVALID : specFast;
    #endif
}

// Tests 20, 23, 24, 27, 28, 54, 59, 65, 66, 76, 81, 98, 112, 117, 124, 126, 128, 134
// TODO: potentially do color clamping after reconstruction in a separate pass

// =====================================================================================
// [注解] 主 Compute Shader 入口
//
// 线程组尺寸来自 `REBLUR_HistoryFix.resources.hlsli`：
//   GROUP_X = 8
//   GROUP_Y = 16
//
// 这里使用 `NRD_CTA_ORDER_REVERSED`，和 PrePass / HistoryFix / PostBlur 一致，
// 属于 NRD 内部的线程组织策略，不改变算法语义。
// =====================================================================================
[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_REVERSED;

    // =====================================================================================
    // [注解] preload 阶段
    //
    // `gIn_Tiles` 是 16x16 tile 分类结果：
    //   - 非 0 表示 sky / 无需处理
    // `PRELOAD_INTO_SMEM_WITH_TILE_CHECK` 会负责把 halo 区一起拉进 shared memory，
    // 并在需要时同步线程组。
    // =====================================================================================
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // [注解] Tile-based early out：天空 tile 或越界线程直接退出
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // [注解] 像素级 early out：超出 denoising range 的像素不参与 HistoryFix
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pixelPos ) ] );
    if( viewZ > gDenoisingRange )
        return;

    // =====================================================================================
    // [注解] 中心像素几何属性
    //
    // 这些量主要用于“短历史重建”阶段的 edge-stopping 权重：
    //   - 法线一致性
    //   - 几何平面距离
    //   - 材质 ID 一致性
    //   - specular 额外的 roughness / hit distance 一致性
    // =====================================================================================
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ], materialID );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );
    float3 Vv = GetViewVector( Xv, true );
    float NoV = abs( dot( Nv, Vv ) );

    // [注解] 当前线程在 shared memory 中对应的位置（带 halo 偏移）
    int2 smemPos = threadPos + NRD_BORDER;

    // =====================================================================================
    // [注解] 从 `Data1` 恢复历史长度（frameNum）
    //
    // `UnpackData1` 返回的是“实际帧数尺度”的 diffuse/specular accumulation speed：
    //   - `frameNum.x`：diffuse 历史长度
    //   - `frameNum.y`：specular 历史长度
    //
    // `frameNumAvgNorm` 则把它归一到 `[0,1]`，以 `gHistoryFixFrameNum` 为上限：
    //   - 0：非常短的历史
    //   - 1：已经足够长，HistoryFix 影响应当减弱
    // =====================================================================================
    float2 frameNum = UnpackData1( gIn_Data1[ pixelPos ] );
    float invHistoryFixFrameNum = 1.0 / max( gHistoryFixFrameNum, NRD_EPS );
    float2 frameNumAvgNorm = saturate( frameNum * invHistoryFixFrameNum );

    // =====================================================================================
    // [注解] HistoryFix 的“采样步长基线”
    //
    // `gHistoryFixBasePixelStride` / `gHistoryFixAlternatePixelStride` 控制：
    //   修复时隔几个像素取一个 tap。
    //
    // 某些材质（由 `gHistoryFixAlternatePixelStrideMaterialID` 指定）会走 alternate stride，
    // 相当于允许特定材质使用不同的修补密度。
    //
    // 后面的几步很关键：
    //   - `/= 2`：对齐 RELAX 的 frameNum 语义
    //   - `*= 2 / radius`：保持“有效模糊半径（像素尺度）”大致一致
    //   - `*= float2(frameNum < gHistoryFixFrameNum)`：
    //       只有历史长度不够时，HistoryFix 才启用；历史够长就直接让 stride 变成 0
    // =====================================================================================
    float2 stride = materialID == gHistoryFixAlternatePixelStrideMaterialID ? gHistoryFixAlternatePixelStride : gHistoryFixBasePixelStride;
    stride /= 1.0 + 1.0; // to match RELAX, where "frameNum" after "TemporalAccumulation" is "1", not "0"
    stride *= 2.0 / REBLUR_HISTORY_FIX_FILTER_RADIUS; // preserve blur radius in pixels ( default blur radius is 2 taps )
    stride *= float2( frameNum < gHistoryFixFrameNum );

    // =====================================================================================
    // [注解] ===================== Diffuse 分支 =====================
    //
    // 整体流程：
    //   1. 读当前 diffuse
    //   2. 根据历史长度和 hit distance 估计 stride
    //   3. 如果 stride != 0，则做“短历史重建”
    //   4. 更新 fast history
    //   5. 用 fast history 的局部方差做 clamp / anti-firefly
    //   6. 回写最终 diffuse
    // =====================================================================================
    #if( NRD_DIFF )
    {
        REBLUR_TYPE diff = gIn_Diff[ pixelPos ];
        #if( NRD_MODE == SH )
            REBLUR_SH_TYPE diffSh = gIn_DiffSh[ pixelPos ];
        #endif

        // [注解] 非线性累计速度：历史越短，值越大，说明越需要快速响应 / 修补
        float diffNonLinearAccumSpeed = 1.0 / ( 1.0 + frameNum.x );

        // =================================================================================
        // [注解] diffuse 的 hit distance 只用于估计“修补 stride 大小”，不是做完整的反射语义分析
        //
        // 步骤：
        //   1. 先把归一化 hit distance 恢复到真实尺度
        //   2. 通过 `GetHitDistFactor` 估计它相对场景尺度的位置
        //   3. 再把 `hitDist` 恢复成原始归一化存储值，供后续权重函数使用
        // =================================================================================
        float hitDistScale = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings.xyz, 1.0 );
        float hitDist = ExtractHitDist( diff ) * hitDistScale;
        float hitDistFactor = GetHitDistFactor( hitDist, frustumSize );
        hitDist = ExtractHitDist( diff );

        // =================================================================================
        // [注解] 估计 stride between taps
        //
        // 历史短 / hit distance 小时，stride 往往更小，重建更局部；
        // 历史更稳时，stride 会趋近 1 或直接变为 0（完全不重建）。
        //
        // `QuadReadAcrossX/Y` 那段是在 2x2 quad 内把 stride 往更稳定邻居靠齐，
        // 防止相邻 lane 因四舍五入不同而出现明显块状边界。
        // =================================================================================
        float diffStride = stride.x;
        diffStride *= lerp( 0.25 + 0.75 * Math::Sqrt01( hitDistFactor ), 1.0, diffNonLinearAccumSpeed ); // "hitDistFactor" is very noisy and breaks nice patterns
        #ifdef NRD_COMPILER_DXC
            // Adapt to neighbors if they are more stable
            float d10 = QuadReadAcrossX( diffStride );
            float d01 = QuadReadAcrossY( diffStride );

            float avg = ( d10 + d01 + diffStride ) / 3.0;
            diffStride = min( diffStride, avg );
        #endif
        diffStride = round( diffStride );

        // =================================================================================
        // [注解] 短历史重建（History reconstruction）
        //
        // 只有 `diffStride != 0` 时才真正进入：
        //   - 0 表示历史已经足够长，不需要再做这轮修复
        //   - 非 0 表示用稀疏邻域重新构造一个更稳定的 normal history
        //
        // 这段看起来像一个简化版 A-trous / edge-aware spatial reconstruction：
        //   - 邻域形状：半径 2 的十字 / 菱形，跳过中心和四角
        //   - 权重：几何、材质、法线、hit distance
        //   - 可选地再乘上邻居自己的 history length（非 performance mode）
        // =================================================================================
        if( diffStride != 0.0 )
        {
            // Parameters
            float normalWeightParam = GetNormalWeightParam( diffNonLinearAccumSpeed, gLobeAngleFraction );
            float2 geometryWeightParams = GetGeometryWeightParams( gPlaneDistSensitivity, frustumSize, Xv, Nv );
            float2 hitDistanceWeightParams = GetHitDistanceWeightParams( hitDist, diffNonLinearAccumSpeed );

            // [注解] `sumd` 初值不是 1，而是 `1 + frameNum.x`
            //        代表“中心样本已经自带一定历史权重”。
            float sumd = 1.0 + frameNum.x;
            #if( REBLUR_PERFORMANCE_MODE == 1 )
                // [注解] 性能模式下使用更便宜的近似，避免直接把完整 frameNum 当作权重
                sumd = 1.0 + 1.0 / ( 1.0 + gMaxAccumulatedFrameNum ) - diffNonLinearAccumSpeed;
            #endif

            diff *= sumd;
            #if( NRD_MODE == SH )
                diffSh *= sumd;
            #endif

            [unroll]
            for( j = -REBLUR_HISTORY_FIX_FILTER_RADIUS; j <= REBLUR_HISTORY_FIX_FILTER_RADIUS; j++ )
            {
                [unroll]
                for( i = -REBLUR_HISTORY_FIX_FILTER_RADIUS; i <= REBLUR_HISTORY_FIX_FILTER_RADIUS; i++ )
                {
                    // Skip center
                    if( i == 0 && j == 0 )
                        continue;

                    // [注解] 跳过四角，只保留十字/菱形 footprint。
                    //        这样既省开销，又能避免过强的对角混合。
                    if( abs( i ) + abs( j ) == REBLUR_HISTORY_FIX_FILTER_RADIUS * 2 )
                        continue;

                    // Sample uv ( at the pixel center )
                    float2 uv = pixelUv + float2( i, j ) * diffStride * gRectSizeInv;

                    // [注解] 镜像 UV，避免采样走出屏幕后直接浪费 tap
                    uv = MirrorUv( uv );

                    // "uv" to "pos"
                    int2 pos = uv * gRectSize; // "uv" can't be "1"

                    // Fetch data
                    float zs = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pos ) ] );
                    float3 Xvs = Geometry::ReconstructViewPosition( uv, gFrustum, zs, gOrthoMode );

                    float materialIDs;
                    float4 Ns = gIn_Normal_Roughness[ WithRectOrigin( pos ) ];
                    Ns = NRD_FrontEnd_UnpackNormalAndRoughness( Ns, materialIDs );

                    // =================================================================================
                    // [注解] diffuse 邻域权重
                    //
                    // 依次考虑：
                    //   1. 平面距离一致性（防止跨表面）
                    //   2. 材质 ID 一致性
                    //   3. 法线一致性
                    //   4. 深度是否仍在降噪范围内
                    //   5. 可选：邻居历史长度越长，权重越高
                    //   6. 最后再乘上 hit distance 一致性（源码注释称作 A-trous weight）
                    // =================================================================================
                    float angle = Math::AcosApprox( dot( Ns.xyz, N ) );
                    float NoX = dot( Nv, Xvs );

                    float w = ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );
                    w *= CompareMaterials( materialID, materialIDs, gDiffMinMaterial );
                    w *= ComputeExponentialWeight( angle, normalWeightParam, 0.0 );
                    w = zs < gDenoisingRange ? w : 0.0; // |NoX| can be ~0 if "zs" is out of range
                    // gaussian weight is not needed

                    #if( REBLUR_PERFORMANCE_MODE == 0 )
                        // [注解] 非性能模式：邻居如果历史更长，就让它在修补中更“有发言权”
                        w *= 1.0 + UnpackData1( gIn_Data1[ pos ] ).x;
                    #endif

                    REBLUR_TYPE s = gIn_Diff[ pos ];
                    s = Denanify( w, s );

                    // A-trous weight
                    w *= ComputeExponentialWeight( ExtractHitDist( s ), hitDistanceWeightParams.x, hitDistanceWeightParams.y );

                    // Accumulate
                    sumd += w;

                    diff += s * w;
                    #if( NRD_MODE == SH )
                        REBLUR_SH_TYPE sh = gIn_DiffSh[ pos ];
                        sh = Denanify( w, sh );
                        diffSh += sh * w;
                    #endif
                }
            }

            sumd = Math::PositiveRcp( sumd );
            diff *= sumd;
            #if( NRD_MODE == SH )
                diffSh *= sumd;
            #endif
        }

        // =================================================================================
        // [注解] 下面开始进入“fast history / variance clamp”阶段
        //
        // `diffLuma`：修复后的主 history 的亮度
        // 后续会拿它和 fast history、局部 moments 一起比较。
        // =================================================================================
        float diffLuma = GetLuma( diff );

        // =================================================================================
        // [注解] 把 fast history 与修复后的 normal history 混合
        //
        // `f = frameNumAvgNorm.x`：
        //   - 历史越长，越偏向已有 fast history
        //   - 历史越短，越偏向当前刚修补出来的 diffLuma
        //
        // 这样可以让 fast history 既保持敏捷，又不至于和主 history 完全脱节。
        // =================================================================================
        float f = frameNumAvgNorm.x;

        float diffFastCenter = s_DiffLuma[ smemPos.y ][ smemPos.x ];
        diffFastCenter = lerp( diffLuma, diffFastCenter, f );

        gOut_DiffFast[ pixelPos ] = diffFastCenter;

        // =================================================================================
        // [注解] 局部方差统计（moments）
        //
        // 这里会统计两套 moments：
        //   1. `diffFastM1/M2`
        //      - 在 `REBLUR_FAST_HISTORY_CLAMPING_RADIUS` 的局部窗口内统计
        //      - 用于 fast history clamping
        //
        //   2. `diffAntiFireflyM1/M2`
        //      - 在更大的 `NRD_BORDER` 邻域上统计，但跳过中心 3x3
        //      - 用于 anti-firefly
        //
        // 遇到 preload 里标记成 `REBLUR_INVALID` 的邻居时，回退为中心 fast luma，
        // 这样边界/背景不会把方差拉得异常大。
        // =================================================================================
        float diffFastM1 = diffFastCenter;
        float diffFastM2 = diffFastCenter * diffFastCenter;
        float diffAntiFireflyM1 = 0;
        float diffAntiFireflyM2 = 0;

        [unroll]
        for( j = -NRD_BORDER; j <= NRD_BORDER; j++ )
        {
            [unroll]
            for( i = -NRD_BORDER; i <= NRD_BORDER; i++ )
            {
                if( i == 0 && j == 0 )
                    continue;

                int2 pos = smemPos + int2( i, j );

                float d = s_DiffLuma[ pos.y ][ pos.x ];
                d = d == REBLUR_INVALID ? diffFastCenter : d;

                // Variance in 5x5 for fast history
                if( abs( i ) <= REBLUR_FAST_HISTORY_CLAMPING_RADIUS && abs( j ) <= REBLUR_FAST_HISTORY_CLAMPING_RADIUS )
                {
                    diffFastM1 += d;
                    diffFastM2 += d * d;
                }

                // Variance in "NRD_BORDER x NRD_BORDER" skipping central 3x3 for anti-firefly
                if( NRD_SUPPORTS_ANTIFIREFLY && !( abs( i ) <= 1 && abs( j ) <= 1 ) )
                {
                    diffAntiFireflyM1 += d;
                    diffAntiFireflyM2 += d * d;
                }
            }
        }

        // =================================================================================
        // [注解] Anti-firefly
        //
        // 用更大邻域估计亮度分布，再把当前 luma 限制到：
        //   mean ± sigma * REBLUR_ANTI_FIREFLY_SIGMA_SCALE
        //
        // 中心 3x3 被故意排除，是为了让真正的孤立亮点更难“自我证明”。
        // =================================================================================
        if( NRD_SUPPORTS_ANTIFIREFLY && gAntiFirefly )
        {
            float invNorm = 1.0 / ( ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 ) - 3 * 3 ); // -9 samples
            diffAntiFireflyM1 *= invNorm;
            diffAntiFireflyM2 *= invNorm;

            float diffAntiFireflySigma = GetStdDev( diffAntiFireflyM1, diffAntiFireflyM2 ) * REBLUR_ANTI_FIREFLY_SIGMA_SCALE;

            diffLuma = clamp( diffLuma, diffAntiFireflyM1 - diffAntiFireflySigma, diffAntiFireflyM1 + diffAntiFireflySigma );
        }

        // =================================================================================
        // [注解] Clamp to fast history
        //
        // 这一步比 anti-firefly 更“日常”：
        //   - fast history 代表当前局部区域近期的亮度稳定范围
        //   - 如果修补后的主 history 偏离太多，就把它压回 fast history 的 sigma 区间
        //
        // 最后那层 lerp 表示：
        //   历史越长，对 clamp 的依赖越小；
        //   fast history 和 normal history 的最大帧数差异越大，clamp 越有意义。
        // =================================================================================
        {
            float invNorm = 1.0 / ( ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) * ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) );
            diffFastM1 *= invNorm;
            diffFastM2 *= invNorm;

            float diffFastSigma = GetStdDev( diffFastM1, diffFastM2 ) * gFastHistoryClampingSigmaScale;
            float diffLumaClamped = clamp( diffLuma, diffFastM1 - diffFastSigma, diffFastM1 + diffFastSigma );

            diffLuma = lerp( diffLumaClamped, diffLuma, 1.0 / ( 1.0 + float( gMaxFastAccumulatedFrameNum < gMaxAccumulatedFrameNum ) * frameNum.x * 2.0 ) );
        }

        // [注解] Debug 显示模式：直接输出 fast history 的亮度
        #if( REBLUR_SHOW == REBLUR_SHOW_FAST_HISTORY )
            diffLuma = diffFastCenter;
        #endif

        // [注解] 只改 luma，不直接改色相 / 方向
        diff = ChangeLuma( diff, diffLuma );
        #if( NRD_MODE == SH )
            diffSh *= GetLumaScale( length( diffSh ), diffLuma );
        #endif

        // Output
        gOut_Diff[ pixelPos ] = diff;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ pixelPos ] = diffSh;
        #endif
    }
    #endif

    // =====================================================================================
    // [注解] ===================== Specular 分支 =====================
    //
    // 和 diffuse 同结构，但 specular 会多考虑几件事：
    //   - `roughness`
    //   - `GetSpecMagicCurve(roughness)`
    //   - `gIn_SpecHitDistForTracking`
    //   - 邻域 roughness 一致性
    //
    // 这使得 specular 的 HistoryFix 对低粗糙度反射会更保守、更敏感。
    // =====================================================================================
    #if( NRD_SPEC )
    {
        REBLUR_TYPE spec = gIn_Spec[ pixelPos ];
        #if( NRD_MODE == SH )
            REBLUR_SH_TYPE specSh = gIn_SpecSh[ pixelPos ];
        #endif

        // [注解] `smc = GetSpecMagicCurve(roughness)`：
        //        REBLUR 里非常常见的 roughness 非线性调制曲线。
        //        低粗糙度时会让很多操作更保守，高粗糙度时逐渐退化得更像 diffuse。
        float smc = GetSpecMagicCurve( roughness );
        float specNonLinearAccumSpeed = 1.0 / ( 1.0 + frameNum.y );

        // =================================================================================
        // [注解] specular 的 hit distance 处理
        //
        // 普通 spec hit distance：
        //   `ExtractHitDist(spec) * normalization`
        //
        // 但在非 OCCLUSION 模式下，源码会把它与 `gIn_SpecHitDistForTracking` 做插值：
        //   - 低 roughness：更偏向 tracking hit distance（更适合反射跟踪）
        //   - 高 roughness：更偏向原始 hit distance（因为 tracking 的最小值语义会过于激进）
        // =================================================================================
        float hitDistScale = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings.xyz, roughness );
        float hitDist = ExtractHitDist( spec ) * hitDistScale;
        #if( NRD_MODE != OCCLUSION )
            // "gIn_SpecHitDistForTracking" is better for low roughness, but doesn't suit for high roughness ( because it's min )
            hitDist = lerp( gIn_SpecHitDistForTracking[ pixelPos ], hitDist, smc );
        #endif
        float hitDistFactor = GetHitDistFactor( hitDist, frustumSize );
        hitDist = saturate( hitDist / hitDistScale );

        // =================================================================================
        // [注解] specular stride
        //
        // 除了 diffuse 那套“历史长度 / hitDistFactor”调制外，
        // 这里还额外乘了 `lerp(0.25, 1.0, smc)`：
        //   - 低 roughness（smc 小） => stride 更小，更保守
        //   - 高 roughness（smc 大） => stride 更接近基线
        // =================================================================================
        float specStride = stride.y;
        specStride *= lerp( 0.25 + 0.75 * Math::Sqrt01( hitDistFactor ), 1.0, specNonLinearAccumSpeed ); // "hitDistFactor" is very noisy and breaks nice patterns
        specStride *= lerp( 0.25, 1.0, smc ); // hand tuned // TODO: use "lobeRadius"?
        #ifdef NRD_COMPILER_DXC
            // Adapt to neighbors if they are more stable
            float d10 = QuadReadAcrossX( specStride );
            float d01 = QuadReadAcrossY( specStride );

            float avg = ( d10 + d01 + specStride ) / 3.0;
            specStride = min( specStride, avg );
        #endif
        specStride = round( specStride );

        // =================================================================================
        // [注解] specular 短历史重建
        //
        // 与 diffuse 的差别主要在权重：
        //   - 法线一致性更重要
        //   - 粗糙度一致性必须参与
        //   - hit distance 对反射语义更敏感
        // =================================================================================
        if( specStride != 0 )
        {
            // Parameters
            float normalWeightParam = GetNormalWeightParam( specNonLinearAccumSpeed, gLobeAngleFraction, roughness );
            float2 geometryWeightParams = GetGeometryWeightParams( gPlaneDistSensitivity, frustumSize, Xv, Nv );
            float2 hitDistanceWeightParams = GetHitDistanceWeightParams( hitDist, specNonLinearAccumSpeed );
            float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams( roughness * roughness, sqrt( gRoughnessFraction ) );

            float sums = 1.0 + frameNum.y;
            #if( REBLUR_PERFORMANCE_MODE == 1 )
                sums = 1.0 + 1.0 / ( 1.0 + gMaxAccumulatedFrameNum ) - specNonLinearAccumSpeed;
            #endif

            spec *= sums;
            #if( NRD_MODE == SH )
                specSh *= sums;
            #endif

            [unroll]
            for( j = -REBLUR_HISTORY_FIX_FILTER_RADIUS; j <= REBLUR_HISTORY_FIX_FILTER_RADIUS; j++ )
            {
                [unroll]
                for( i = -REBLUR_HISTORY_FIX_FILTER_RADIUS; i <= REBLUR_HISTORY_FIX_FILTER_RADIUS; i++ )
                {
                    // Skip center
                    if( i == 0 && j == 0 )
                        continue;

                    // Skip corners
                    if( abs( i ) + abs( j ) == REBLUR_HISTORY_FIX_FILTER_RADIUS * 2 )
                        continue;

                    // Sample uv ( at the pixel center )
                    float2 uv = pixelUv + float2( i, j ) * specStride * gRectSizeInv;

                    // Apply "mirror" to not waste taps going outside of the screen
                    uv = MirrorUv( uv );

                    // "uv" to "pos"
                    int2 pos = uv * gRectSize; // "uv" can't be "1"

                    // Fetch data
                    float zs = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pos ) ] );
                    float3 Xvs = Geometry::ReconstructViewPosition( uv, gFrustum, zs, gOrthoMode );

                    float materialIDs;
                    float4 Ns = gIn_Normal_Roughness[ WithRectOrigin( pos ) ];
                    Ns = NRD_FrontEnd_UnpackNormalAndRoughness( Ns, materialIDs );

                    // [注解] specular 权重：几何 + 材质 + 法线 + 粗糙度 + hit distance
                    float angle = Math::AcosApprox( dot( Ns.xyz, N ) );
                    float NoX = dot( Nv, Xvs );

                    float w = ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );
                    w *= CompareMaterials( materialID, materialIDs, gSpecMinMaterial );
                    w *= ComputeExponentialWeight( angle, normalWeightParam, 0.0 );
                    w *= ComputeExponentialWeight( Ns.w * Ns.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y );
                    w = zs < gDenoisingRange ? w : 0.0; // |NoX| can be ~0 if "zs" is out of range
                    // gaussian weight is not needed

                    #if( REBLUR_PERFORMANCE_MODE == 0 )
                        w *= 1.0 + UnpackData1( gIn_Data1[ pos ] ).y;
                    #endif

                    REBLUR_TYPE s = gIn_Spec[ pos ];
                    s = Denanify( w, s );

                    // A-trous weight
                    w *= ComputeExponentialWeight( ExtractHitDist( s ), hitDistanceWeightParams.x, hitDistanceWeightParams.y );

                    // Accumulate
                    sums += w;

                    spec += s * w;
                    #if( NRD_MODE == SH )
                        REBLUR_SH_TYPE sh = gIn_SpecSh[ pos ];
                        sh = Denanify( w, sh );
                        specSh += sh * w;
                    #endif
                }
            }

            sums = Math::PositiveRcp( sums );
            spec *= sums;
            #if( NRD_MODE == SH )
                specSh *= sums;
            #endif
        }

        float specLuma = GetLuma( spec );

        // =================================================================================
        // [注解] 更新 spec fast history
        //
        // 和 diffuse 类似，但这里多了一层：
        //   `f = lerp(1, frameNumAvgNorm.y, smc)`
        //
        // 含义：
        //   - 低 roughness（smc 小）时，更不希望把 HistoryFix 过的结果写入 fast history
        //     因为镜面反射对错误的空间修补非常敏感
        //   - 高 roughness 时，行为逐渐接近 diffuse
        // =================================================================================
        float f = frameNumAvgNorm.y;
        f = lerp( 1.0, f, smc ); // HistoryFix-ed data is undesired in fast history for low roughness ( test 115 )

        float specFastCenter = s_SpecLuma[ smemPos.y ][ smemPos.x ];
        specFastCenter = lerp( specLuma, specFastCenter, f );

        gOut_SpecFast[ pixelPos ] = specFastCenter;

        // Local variance
        float specFastM1 = specFastCenter;
        float specFastM2 = specFastCenter * specFastCenter;
        float specAntiFireflyM1 = 0;
        float specAntiFireflyM2 = 0;

        [unroll]
        for( j = -NRD_BORDER; j <= NRD_BORDER; j++ )
        {
            [unroll]
            for( i = -NRD_BORDER; i <= NRD_BORDER; i++ )
            {
                if( i == 0 && j == 0 )
                    continue;

                int2 pos = smemPos + int2( i, j );

                float s = s_SpecLuma[ pos.y ][ pos.x ];
                s = s == REBLUR_INVALID ? specFastCenter : s;

                // Variance in 5x5 for fast history
                if( abs( i ) <= REBLUR_FAST_HISTORY_CLAMPING_RADIUS && abs( j ) <= REBLUR_FAST_HISTORY_CLAMPING_RADIUS )
                {
                    specFastM1 += s;
                    specFastM2 += s * s;
                }

                // Variance in "NRD_BORDER x NRD_BORDER" skipping central 3x3 for anti-firefly
                if( NRD_SUPPORTS_ANTIFIREFLY && !( abs( i ) <= 1 && abs( j ) <= 1 ) )
                {
                    specAntiFireflyM1 += s;
                    specAntiFireflyM2 += s * s;
                }
            }
        }

        // Anti-firefly
        if( NRD_SUPPORTS_ANTIFIREFLY && gAntiFirefly )
        {
            float invNorm = 1.0 / ( ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 ) - 3 * 3 ); // -9 samples
            specAntiFireflyM1 *= invNorm;
            specAntiFireflyM2 *= invNorm;

            float specAntiFireflySigma = GetStdDev( specAntiFireflyM1, specAntiFireflyM2 ) * REBLUR_ANTI_FIREFLY_SIGMA_SCALE;

            specLuma = clamp( specLuma, specAntiFireflyM1 - specAntiFireflySigma, specAntiFireflyM1 + specAntiFireflySigma );
        }

        // =================================================================================
        // [注解] specular fast history clamp
        //
        // 与 diffuse 基本一致，但 strand 材质会额外放宽 sigma：
        //   因为头发/纤维类材质的局部亮度变化天然更剧烈，
        //   如果 clamp 太严，容易过度压平闪烁细节。
        // =================================================================================
        {
            float invNorm = 1.0 / ( ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) * ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) );
            specFastM1 *= invNorm;
            specFastM2 *= invNorm;

            float fastHistoryClampingSigmaScale = gFastHistoryClampingSigmaScale;
            if( materialID == gStrandMaterialID )
                fastHistoryClampingSigmaScale = max( fastHistoryClampingSigmaScale, 3.0 );

            float specFastSigma = GetStdDev( specFastM1, specFastM2 ) * fastHistoryClampingSigmaScale;
            float specLumaClamped = clamp( specLuma, specFastM1 - specFastSigma, specFastM1 + specFastSigma );

            specLuma = lerp( specLumaClamped, specLuma, 1.0 / ( 1.0 + float( gMaxFastAccumulatedFrameNum < gMaxAccumulatedFrameNum ) * frameNum.y * 2.0 ) );
        }

        // Change luma
        #if( REBLUR_SHOW == REBLUR_SHOW_FAST_HISTORY )
            specLuma = specFastCenter;
        #endif

        spec = ChangeLuma( spec, specLuma );
        #if( NRD_MODE == SH )
            specSh *= GetLumaScale( length( specSh ), specLuma );
        #endif

        // Output
        gOut_Spec[ pixelPos ] = spec;
        #if( NRD_MODE == SH )
            gOut_SpecSh[ pixelPos ] = specSh;
        #endif
    }
    #endif
}
