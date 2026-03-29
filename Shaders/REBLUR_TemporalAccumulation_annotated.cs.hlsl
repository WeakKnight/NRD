/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR Temporal Accumulation（时域累积）
//
// 它是 REBLUR 管线里最关键的时域融合 pass，位于：
//   Pre-pass / HitDistReconstruction 之后
//   History fix / Blur / Post-blur 之前
//
// 主要职责可以分成三层：
//   1. **Surface Motion Based Reprojection（SMB）**
//      - 用常规运动矢量把当前像素投回上一帧表面
//      - 检查上一帧深度 / 法线 / 材质是否仍然匹配当前表面
//
//   2. **Virtual Motion Based Reprojection（VMB）**（仅 Specular）
//      - 针对镜面/半镜面反射，单纯的表面运动并不足以描述反射点的真实漂移
//      - 因此会结合 hit distance、粗糙度、曲率估计构造“虚拟反射点”
//      - 再用这个虚拟点去上一帧取历史，减少高光/反射拖影
//
//   3. **History Mixing + Reliability Control**
//      - 依据 surface / virtual 两条历史路径各自的置信度、足迹质量、checkerboard 状态
//        决定混多少历史、混哪条历史，以及对累积帧数施加怎样的上限
//      - 同时输出供后续 pass 使用的 Data1 / Data2 / FastHistory / HitDistForTracking
//
// 本文件和 `REBLUR_PrePass_annotated.cs.hlsl`、`REBLUR_Common_SpatialFilter_annotated.hlsli`
// 是强耦合的：
//   - Pre-pass 负责提供更稳定的 specular hit distance for tracking
//   - TemporalAccumulation 把当前帧与历史帧合并
//   - 后续 Blur / HistoryFix / TemporalStabilization 会继续消费这里产出的数据
// =====================================================================================

// [注解] NRD 核心头文件：导出宏、基础类型、公共常量
#include "NRD.hlsli"

// [注解] 数学库：近似 acos、smoothstep、pow、saturate 等
#include "ml.hlsli"

// [注解] REBLUR 全局配置：
//        包括 CatRom / STF / firefly suppressor / virtual motion 等开关和阈值
#include "REBLUR_Config.hlsli"

// [注解] 当前 pass 的资源绑定声明：
//        输入包括当前帧 normal/viewZ/motion vectors、上一帧 history / internal data / prev viewZ 等
//        输出包括 gOut_Data1 / gOut_Data2 / gOut_Diff / gOut_Spec / FastHistory / SH / hitDistForTracking
#include "REBLUR_TemporalAccumulation.resources.hlsli"

// [注解] 通用工具：坐标变换、CatRom/Bilinear filter、packing 等
#include "Common.hlsli"

// [注解] REBLUR 公共函数：
//        PackData1 / PackData2 / MixHistoryAndCurrent / hit distance normalization 等均定义在这里
#include "REBLUR_Common.hlsli"

// =====================================================================================
// [注解] groupshared 缓存
//
// TemporalAccumulation 只把“局部、稳定、可重复读取”的数据塞进 shared memory：
//   - `s_Normal_Roughness`：邻域法线与粗糙度
//   - `s_HitDistForTracking`：specular 用于追踪的 hit distance（无效值存成 NRD_INF，便于取最小值）
//
// 这样做的目的：
//   - 当前像素会在 3x3 邻域内统计平均法线和粗糙度方差
//   - 这些读取是规整访问，适合 shared memory
//   - 真正的 reprojection / history sampling 仍然必须访问全局纹理，因为坐标是连续/稀疏的
// =====================================================================================
groupshared float4 s_Normal_Roughness[ BUFFER_Y ][ BUFFER_X ];
groupshared float s_HitDistForTracking[ BUFFER_Y ][ BUFFER_X ];

// =====================================================================================
// [注解] StochasticBilinear
//
// 这是一个“小工具函数”：
//   - 默认直接返回原始 uv
//   - 只有在 `REBLUR_USE_STF == 1` 且法线编码是 `R10G10B10A2` 时，
//     才启用随机双线性采样（STF = stochastic texture filtering）
//
// 作用：
//   - 对低比特法线/粗糙度纹理，直接线性采样有时会出现固定模式偏差
//   - STF 通过随机选择双线性 footprint 内的离散 texel，
//     让结果更接近统计意义上的线性采样，轻微改善调试可视化下的质量
// =====================================================================================
float2 StochasticBilinear( float2 uv, float2 texSize )
{
    #if( REBLUR_USE_STF == 1 && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        // [注解] 先构造标准 bilinear footprint（origin + 4 个权重）
        //        注意：调用方必须先初始化哈希 RNG
        Filtering::Bilinear f = Filtering::GetBilinearFilter( uv, texSize );

        // [注解] 随机数落在 bilinear 权重累积分布上，决定向哪个离散 texel 偏移
        float2 rnd = Rng::Hash::GetFloat2( );
        f.origin += step( rnd, f.weights );

        // [注解] 返回被随机选中的像素中心坐标
        return ( f.origin + 0.5 ) / texSize;
    #else
        // [注解] 大多数情况下直接使用原始 uv
        return uv;
    #endif
}

// =====================================================================================
// [注解] Preload：把规则邻域会反复访问的数据装入 shared memory
//
// 与 `REBLUR_HitDistReconstruction_annotated.cs.hlsl` 的 preload 思路类似，
// 但这里缓存的内容更少：只缓存“法线/粗糙度”和“specular tracking hit distance”。
//
// 特别注意 `s_HitDistForTracking`：
//   - 仅在 `NRD_SPEC` 编译分支下存在意义
//   - 如果 pre-pass 关闭（`gSpecPrepassBlurRadius == 0`），tracking hit distance 直接来自当前 spec 输入
//   - 如果 pre-pass 开启，则优先读取 pre-pass 输出的 `gIn_SpecHitDistForTracking`
//   - 无效样本统一写成 `NRD_INF`，便于后续做最小值归约
// =====================================================================================
void Preload( uint2 sharedPos, int2 globalPos )
{
    // [注解] clamp 到有效屏幕区域，避免 halo 访问越界
    globalPos = clamp( globalPos, 0, gRectSizeMinusOne );

    // [注解] 缓存当前像素的法线 + 粗糙度
    s_Normal_Roughness[ sharedPos.y ][ sharedPos.x ] = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( globalPos ) ] );

    #if( NRD_SPEC )
        #if( NRD_MODE == OCCLUSION )
            // [注解] Occlusion 模式下，checkerboard 的 spec 输入可能是半宽纹理
            uint shift = gSpecCheckerboard != 2 ? 1 : 0;
            uint2 pos = uint2( globalPos.x >> shift, globalPos.y );
        #else
            uint2 pos = globalPos;
        #endif

        REBLUR_TYPE spec = gIn_Spec[ pos ];
        #if( NRD_MODE == OCCLUSION )
            // [注解] Occlusion 模式没有 pre-pass tracking 纹理，直接取输入 hit distance
            float hitDist = ExtractHitDist( spec );
        #else
            // [注解] 如果 pre-pass 关闭，就用当前 spec 的 hit distance；
            //        否则使用 pre-pass 产出的 tracking hit distance
            float hitDist = gSpecPrepassBlurRadius == 0.0 ? ExtractHitDist( spec ) : gIn_SpecHitDistForTracking[ globalPos ];
        #endif

        float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( globalPos ) ] );

        // [注解] 仅对“有效且在降噪范围内”的样本保留 hit distance，
        //        其余写成 NRD_INF，后续可以直接做 min() 归约
        s_HitDistForTracking[ sharedPos.y ][ sharedPos.x ] = ( hitDist == 0.0 || viewZ > gDenoisingRange ) ? NRD_INF : hitDist; // for "min"
    #endif
}

// =====================================================================================
// [注解] 主 Compute Shader 入口
//
// TemporalAccumulation 使用和 HitDistReconstruction 相同的 CTA 排布方式：
//   - `NRD_CTA_ORDER_DEFAULT` 负责生成 `pixelPos` / `threadPos`
//   - 线程组尺寸来自 `REBLUR_TemporalAccumulation.resources.hlsli`
// =====================================================================================
[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_DEFAULT;

    // =====================================================================================
    // [注解] Preload 阶段
    //
    // `gIn_Tiles` 是 16x16 的 tile 分类结果：
    //   - 非 0 表示该 tile 是天空 / 无需降噪
    // `PRELOAD_INTO_SMEM_WITH_TILE_CHECK` 会：
    //   - 读取 halo 区域到 shared memory
    //   - 做必要同步
    //   - 天空 tile 会尽早跳过
    // =====================================================================================
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // [注解] tile 级 early out：天空或越界线程直接退出
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // [注解] 像素级 early out：超出降噪范围的像素不参与 temporal accumulation
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pixelPos ) ] );
    if( viewZ > gDenoisingRange )
        return;

    // =====================================================================================
    // [注解] 当前帧像素位置重建
    //
    // `pixelUv`：像素中心的归一化屏幕坐标
    // `Xv`     ：当前像素的观察空间位置
    // `X`      ：当前像素的世界空间位置
    //
    // 后面 Surface Motion / Virtual Motion 都会围绕 `X` 与 `Xprev` 展开
    // =====================================================================================
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );
    float3 X = Geometry::RotateVector( gViewToWorld, Xv );

    // =====================================================================================
    // [注解] 局部 3x3 邻域统计
    //
    // 这里做两类统计：
    //   - `Navg`：中心 2x2 小邻域的平均法线（保持未归一化）
    //   - `hitDistForTracking` / `roughnessM1/M2`：仅 specular 路径使用
    //
    // 这些统计分别服务于：
    //   - 修正后的 roughness（基于法线方差）
    //   - virtual motion 的 hit distance / roughness 置信度估计
    // =====================================================================================
    float3 Navg = 0.0;
    #if( NRD_SPEC )
        float hitDistForTracking = NRD_INF;
        float roughnessM1 = 0.0;
        float roughnessM2 = 0.0;
    #endif

    [unroll]
    for( j = 0; j <= NRD_BORDER * 2; j++ )
    {
        [unroll]
        for( i = 0; i <= NRD_BORDER * 2; i++ )
        {
            int2 pos = threadPos + int2( i, j );
            float4 normalAndRoughness = s_Normal_Roughness[ pos.y ][ pos.x ];

            // [注解] 只对中心 2x2 footprint 求平均法线，
            //        因为后续 surface motion reprojection 最终也是 2x2 / bilinear footprint
            if( i < 2 && j < 2 )
                Navg += normalAndRoughness.xyz;

            #if( NRD_SPEC )
                // [注解] tracking hit distance 取局部最小值：
                //        最近的有效反射命中点通常更适合作为 specular motion 的跟踪参考
                //        0 值已在 preload 时映射为 NRD_INF，因此可直接 min()
                float h = s_HitDistForTracking[ pos.y ][ pos.x ];
                hitDistForTracking = min( hitDistForTracking, h );

                // [注解] 估计粗糙度方差时使用 roughness^2，
                //        因为后续测试与 BRDF lobe 宽度更接近 roughness 的平方空间
                float roughnessSq = normalAndRoughness.w * normalAndRoughness.w;
                roughnessM1 += roughnessSq;
                roughnessM2 += roughnessSq * roughnessSq;
            #endif
        }
    }

    // [注解] 保持未归一化的 2x2 法线和，这正是 `GetModifiedRoughnessFromNormalVariance` 需要的输入形式
    Navg /= 4.0; // needs to be unnormalized!

    // =====================================================================================
    // [注解] 中心像素材质属性
    //
    // 这里重新从全局纹理读取中心像素，而不是直接从 shared memory 取，
    // 原因是当前像素还需要材质 ID，而 preload 只缓存了 normal + roughness
    // =====================================================================================
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ], materialID );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    #if( NRD_SPEC )
        // [注解] 根据法线方差修正 roughness：
        //        邻域法线越不一致，说明局部高频几何越强，等效 roughness 会被抬高，
        //        使 temporal mixing 更保守、更稳定
        float roughnessModified = Filtering::GetModifiedRoughnessFromNormalVariance( roughness, Navg ); // TODO: needed?

        roughnessM1 /= ( 1 + NRD_BORDER * 2 ) * ( 1 + NRD_BORDER * 2 );
        roughnessM2 /= ( 1 + NRD_BORDER * 2 ) * ( 1 + NRD_BORDER * 2 );
        float roughnessSigma = GetStdDev( roughnessM1, roughnessM2 );
    #endif

    // =====================================================================================
    // [注解] 输出 specular tracking hit distance
    //
    // 这里的行为需要和 Pre-pass 交叉理解：
    //   - 若 pre-pass 关闭，则当前 pass 负责把 hit distance 从“归一化存储空间”恢复到真实尺度
    //   - 若 pre-pass 已开启，则 `gIn_SpecHitDistForTracking` 已经是 pre-pass 产出的 tracking 距离，
    //     此处不再重复乘 `hitDistNormalization`
    //
    // 这样做能避免同一份 tracking 数据被归一化两次
    // =====================================================================================
    #if( NRD_SPEC )
        #if( REBLUR_USE_STF == 1 && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
            // [注解] STF 依赖每像素初始化的哈希 RNG
            Rng::Hash::Initialize( pixelPos, gFrameIndex );
        #endif

        hitDistForTracking = hitDistForTracking == NRD_INF ? 0.0 : hitDistForTracking;

        float hitDistNormalization = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings.xyz, roughness );
        #if( NRD_MODE == OCCLUSION )
            hitDistForTracking *= hitDistNormalization;
        #else
            hitDistForTracking *= gSpecPrepassBlurRadius == 0.0 ? hitDistNormalization : 1.0;
        #endif

        // [注解] 无论后续最终选择 SMB 还是 VMB，这份 tracking hit distance 都会被后续 pass 继续消费
        gOut_SpecHitDistForTracking[ pixelPos ] = hitDistForTracking;
    #endif

    // =====================================================================================
    // [注解] Surface Motion（SMB）对应的上一帧坐标重建
    //
    // `mv` 是输入运动矢量，经 `gMvScale.xyz` 缩放后可表示屏幕空间偏移 / 世界空间偏移的不同编码语义。
    // 这里存在两种路径：
    //   1. `gMvScale.w == 0`：motion vector 不直接表示上一帧世界坐标，需要借助 `viewZprev` 重建
    //   2. 否则：把 `mv` 当作世界空间偏移直接加到 `X`
    //
    // 最终得到：
    //   - `Xprev`     ：当前点在上一帧对应的世界位置估计
    //   - `smbPixelUv`: 该位置投影到上一帧屏幕后的 uv
    // =====================================================================================
    float3 mv = gIn_Mv[ WithRectOrigin( pixelPos ) ] * gMvScale.xyz;
    float3 Xprev = X;
    float2 smbPixelUv = pixelUv + mv.xy;

    if( gMvScale.w == 0.0 )
    {
        // [注解] 某些 MV 编码不提供 z，需要通过当前世界点转换到 prev view 再和当前 viewZ 求差补出来
        if( gMvScale.z == 0.0 )
            mv.z = Geometry::AffineTransform( gWorldToViewPrev, X ).z - viewZ;

        float viewZprev = viewZ + mv.z;
        float3 Xvprevlocal = Geometry::ReconstructViewPosition( smbPixelUv, gFrustumPrev, viewZprev, gOrthoMode ); // TODO: use gOrthoModePrev

        Xprev = Geometry::RotateVectorInverse( gWorldToViewPrev, Xvprevlocal ) + gCameraDelta.xyz;
    }
    else
    {
        // [注解] 另一种编码下 MV 直接表示世界空间偏移
        Xprev += mv;
        smbPixelUv = Geometry::GetScreenUv( gWorldToClipPrev, Xprev );
    }

    // =====================================================================================
    // [注解] 上一帧 surface motion footprint 的深度收集
    //
    // 这里同时准备两种 footprint：
    //   - `smbCatromFilter`：用于允许 CatRom 时的 4x4 gather
    //   - `smbBilinearFilter`：用于始终存在的 2x2 双线性 footprint
    //
    // 注释图解释了为什么 gather 返回的 4x4 排列需要重新映射成 CatRom 和 bilinear 都能消费的顺序。
    // =====================================================================================
    /*
          Gather      => CatRom12    => Bilinear
        0x 0y 1x 1y       0y 1x
        0z 0w 1z 1w    0z 0w 1z 1w       0w 1z
        2x 2y 3x 3y    2x 2y 3x 3y       2y 3x
        2z 2w 3z 3w       2w 3z

         CatRom12     => Bilinear
           0x 1x
        0y 0z 1y 1z       0z 1y
        2x 2y 3x 3y       2y 3x
           2z 3z
    */
    Filtering::CatmullRom smbCatromFilter = Filtering::GetCatmullRomFilter( smbPixelUv, gRectSizePrev );
    float2 smbCatromGatherUv = smbCatromFilter.origin * gResourceSizeInvPrev;
    float4 smbViewZ0 = gPrev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 1 ) ).wzxy;
    float4 smbViewZ1 = gPrev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 1 ) ).wzxy;
    float4 smbViewZ2 = gPrev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 3 ) ).wzxy;
    float4 smbViewZ3 = gPrev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 3 ) ).wzxy;

    float3 prevViewZ0 = UnpackViewZ( smbViewZ0.yzw );
    float3 prevViewZ1 = UnpackViewZ( smbViewZ1.xzw );
    float3 prevViewZ2 = UnpackViewZ( smbViewZ2.xyw );
    float3 prevViewZ3 = UnpackViewZ( smbViewZ3.xyz );

    // =====================================================================================
    // [注解] 估计 surface motion footprint 对应的上一帧平均法线
    //
    // 之所以不是只取一个上一帧法线，而是取 2x2 footprint 的平均：
    //   - surface motion reprojection 最终落点通常在双线性 footprint 内
    //   - 平均法线对轻微 jitter / footprint 拉伸更鲁棒
    //
    // 注：代码里明确提到 bilinear footprint 可能触及 sky，因此 post-blur 会给 sky 像素写特殊值
    // =====================================================================================
    Filtering::Bilinear smbBilinearFilter = Filtering::GetBilinearFilter( smbPixelUv, gRectSizePrev );
    float3 smbNavg = 0; // TODO: the sum works well, but probably there is a potential to use just "1 smart sample" (see test 168)
    {
        uint2 p = uint2( smbBilinearFilter.origin );
        float sum = 0.0;

        float w = float( prevViewZ0.z < gDenoisingRange );
        smbNavg = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness[ p ] ).xyz * w;
        sum += w;

        w = float( prevViewZ1.y < gDenoisingRange );
        smbNavg += NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness[ p + uint2( 1, 0 ) ] ).xyz * w;
        sum += w;

        w = float( prevViewZ2.y < gDenoisingRange );
        smbNavg += NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness[ p + uint2( 0, 1 ) ] ).xyz * w;
        sum += w;

        w = float( prevViewZ3.x < gDenoisingRange );
        smbNavg += NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness[ p + uint2( 1, 1 ) ] ).xyz * w;
        sum += w;


        smbNavg /= sum == 0.0 ? 1.0 : sum;
    }
    smbNavg = Geometry::RotateVector( gWorldPrevToWorld, smbNavg );

    // =====================================================================================
    // [注解] Surface Motion 的视差估计（Parallax）
    //
    // 这里从正反两个方向估计当前像素在时域上的投影位移：
    //   - `smbParallaxInPixelsMax` 用来控制反遮挡测试的严格程度
    //   - `smbParallaxInPixelsMin` 用来给高视差曲率修正提供一个更保守的尺度
    //
    // 直觉上：
    //   视差越大，纯 surface motion 越容易失真，后续就越需要依赖更严格的筛选或转向 virtual motion
    // =====================================================================================
    // Parallax

    float smbParallaxInPixels1 = ComputeParallaxInPixels( Xprev + gCameraDelta.xyz, gOrthoMode == 0.0 ? smbPixelUv : pixelUv, gWorldToClipPrev, gRectSize );
    float smbParallaxInPixels2 = ComputeParallaxInPixels( Xprev - gCameraDelta.xyz, gOrthoMode == 0.0 ? pixelUv : smbPixelUv, gWorldToClip, gRectSize );

    float smbParallaxInPixelsMax = max( smbParallaxInPixels1, smbParallaxInPixels2 );
    float smbParallaxInPixelsMin = min( smbParallaxInPixels1, smbParallaxInPixels2 );

    // =====================================================================================
    // [注解] 反遮挡阈值（Disocclusion Threshold）
    //
    // 这里会综合：
    //   - 全局阈值 `gDisocclusionThreshold`
    //   - 可选的按像素混合阈值 `gIn_DisocclusionThresholdMix`
    //   - Strand 材质的专门放宽逻辑
    //   - 视差大小与视角（NoV）
    //
    // 目的不是“尽量多复用历史”，而是“只在足够可信时才复用历史”。
    // =====================================================================================
    // Disocclusion: threshold

    float pixelSize = PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ );
    float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );

    float disocclusionThresholdMix = 0;
    if( materialID == gStrandMaterialID )
        disocclusionThresholdMix = NRD_GetNormalizedStrandThickness( gStrandThickness, pixelSize );
    if( gHasDisocclusionThresholdMix && NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX )
        disocclusionThresholdMix = gIn_DisocclusionThresholdMix[ pixelPos ];

    float disocclusionThreshold = lerp( gDisocclusionThreshold, gDisocclusionThresholdAlternate, disocclusionThresholdMix );
    if( materialID == gStrandMaterialID )
    {
        // Further relax "disocclusionThreshold" if parallax is relatively small
        float mediumParallax = Math::SmoothStep01( smbParallaxInPixelsMax );
        disocclusionThreshold = lerp( NRD_STRAND_RELAXED_DISOCCLUSION_THRESHOLD, disocclusionThreshold, mediumParallax );
    }

    // TODO: small parallax ( very slow motion ) could be used to increase disocclusion threshold, but:
    // - MVs should be dilated first
    // - Problem: a static pixel ( with relaxed threshold ) can touch a moving pixel, leading to reprojection artefacts
    float smallParallax = Math::LinearStep( 0.25, 0.0, smbParallaxInPixelsMax );
    float thresholdAngle = REBLUR_ALMOST_ZERO_ANGLE - 0.25 * smallParallax;

    float3 V = GetViewVector( X );
    float NoV = abs( dot( N, V ) );
    float NoVstrict = lerp( NoV, 1.0, saturate( smbParallaxInPixelsMax / 30.0 ) );
    float4 smbDisocclusionThreshold = GetDisocclusionThreshold( disocclusionThreshold, frustumSize, NoVstrict );
    smbDisocclusionThreshold *= float( dot( smbNavg, Navg ) > thresholdAngle ); // good for smb
    smbDisocclusionThreshold *= IsInScreenBilinear( smbBilinearFilter.origin, gRectSizePrev );
    smbDisocclusionThreshold -= NRD_EPS;

    // =====================================================================================
    // [注解] SMB 反遮挡测试：平面距离一致性
    //
    // `Xprev` 是“当前像素在上一帧对应的表面位置估计”。
    // 这里把上一帧 footprint 的深度与 `Xprev.z` 比较，如果偏差超出阈值，就判定该 tap 不可信。
    //
    // 这一步是 SMB 历史是否还能沿用的第一道硬门槛。
    // =====================================================================================
    // Disocclusion: plane distance
    float3 Xvprev = Geometry::AffineTransform( gWorldToViewPrev, Xprev );

    float3 smbPlaneDist0 = abs( prevViewZ0 - Xvprev.z );
    float3 smbPlaneDist1 = abs( prevViewZ1 - Xvprev.z );
    float3 smbPlaneDist2 = abs( prevViewZ2 - Xvprev.z );
    float3 smbPlaneDist3 = abs( prevViewZ3 - Xvprev.z );
    float3 smbOcclusion0 = step( smbPlaneDist0, smbDisocclusionThreshold.x );
    float3 smbOcclusion1 = step( smbPlaneDist1, smbDisocclusionThreshold.y );
    float3 smbOcclusion2 = step( smbPlaneDist2, smbDisocclusionThreshold.z );
    float3 smbOcclusion3 = step( smbPlaneDist3, smbDisocclusionThreshold.w );

    // Disocclusion: materialID
    #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        uint4 smbInternalData0 = gPrev_InternalData.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 1 ) ).wzxy;
        uint4 smbInternalData1 = gPrev_InternalData.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 1 ) ).wzxy;
        uint4 smbInternalData2 = gPrev_InternalData.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 3 ) ).wzxy;
        uint4 smbInternalData3 = gPrev_InternalData.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 3 ) ).wzxy;

        float3 smbMaterialID0 = float3( UnpackInternalData( smbInternalData0.y ).z, UnpackInternalData( smbInternalData0.z ).z, UnpackInternalData( smbInternalData0.w ).z );
        float3 smbMaterialID1 = float3( UnpackInternalData( smbInternalData1.x ).z, UnpackInternalData( smbInternalData1.z ).z, UnpackInternalData( smbInternalData1.w ).z );
        float3 smbMaterialID2 = float3( UnpackInternalData( smbInternalData2.x ).z, UnpackInternalData( smbInternalData2.y ).z, UnpackInternalData( smbInternalData2.w ).z );
        float3 smbMaterialID3 = float3( UnpackInternalData( smbInternalData3.x ).z, UnpackInternalData( smbInternalData3.y ).z, UnpackInternalData( smbInternalData3.z ).z );

        float minMaterialID = min( gSpecMinMaterial, gDiffMinMaterial ); // TODO: separation is expensive
        smbOcclusion0 *= CompareMaterials( materialID, smbMaterialID0, minMaterialID );
        smbOcclusion1 *= CompareMaterials( materialID, smbMaterialID1, minMaterialID );
        smbOcclusion2 *= CompareMaterials( materialID, smbMaterialID2, minMaterialID );
        smbOcclusion3 *= CompareMaterials( materialID, smbMaterialID3, minMaterialID );

        uint4 smbInternalData = uint4( smbInternalData0.w, smbInternalData1.z, smbInternalData2.y, smbInternalData3.x );
    #else
        float2 smbBilinearGatherUv = ( smbBilinearFilter.origin + 1.0 ) * gResourceSizeInvPrev;
        uint4 smbInternalData = gPrev_InternalData.GatherRed( gNearestClamp, smbBilinearGatherUv ).wzxy;
    #endif

    // =====================================================================================
    // [注解] SMB footprint 权重与历史长度恢复
    //
    // `gPrev_InternalData` 在 `REBLUR_Common.hlsli` 中由 `PackInternalData` 打包，包含：
    //   - diffuse accumulation speed
    //   - specular accumulation speed
    //   - material ID
    //
    // TemporalAccumulation 在这里把它解包回来，并按当前 footprint 的可用性做插值，
    // 得到这条 SMB 路径下可继承的历史长度。
    // =====================================================================================
    // 2x2 occlusion weights

    float4 smbOcclusionWeights = Filtering::GetBilinearCustomWeights( smbBilinearFilter, float4( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x ) );
    bool smbAllowCatRom = dot( smbOcclusion0 + smbOcclusion1 + smbOcclusion2 + smbOcclusion3, 1.0 ) > 11.5 && REBLUR_USE_CATROM_FOR_SURFACE_MOTION_IN_TA;

    // Save disocclusion bits
    float fbits = smbOcclusion0.z * 1.0;
    fbits += smbOcclusion1.y * 2.0;
    fbits += smbOcclusion2.y * 4.0;
    fbits += smbOcclusion3.x * 8.0;

    // Accumulation speed
    float2 internalData00 = UnpackInternalData( smbInternalData.x ).xy;
    float2 internalData10 = UnpackInternalData( smbInternalData.y ).xy;
    float2 internalData01 = UnpackInternalData( smbInternalData.z ).xy;
    float2 internalData11 = UnpackInternalData( smbInternalData.w ).xy;

    #if( NRD_DIFF )
        float4 diffAccumSpeeds = float4( internalData00.x, internalData10.x, internalData01.x, internalData11.x );
        float diffAccumSpeed = Filtering::ApplyBilinearCustomWeights( diffAccumSpeeds.x, diffAccumSpeeds.y, diffAccumSpeeds.z, diffAccumSpeeds.w, smbOcclusionWeights );
    #endif

    #if( NRD_SPEC )
        float4 specAccumSpeeds = float4( internalData00.y, internalData10.y, internalData01.y, internalData11.y );
        float smbSpecAccumSpeed = Filtering::ApplyBilinearCustomWeights( specAccumSpeeds.x, specAccumSpeeds.y, specAccumSpeeds.z, specAccumSpeeds.w, smbOcclusionWeights );
    #endif

    // Footprint quality
    float3 smbVprev = GetViewVectorPrev( Xprev, gCameraDelta.xyz );
    float NoVprev = abs( dot( N, smbVprev ) ); // TODO: should be smbNavg ( normalized? ), but jittering breaks logic
    float sizeQuality = ( NoVprev + 1e-3 ) / ( NoV + 1e-3 ); // this order because we need to fix stretching only, shrinking is OK
    sizeQuality *= sizeQuality;
    sizeQuality = lerp( 0.1, 1.0, saturate( sizeQuality ) );

    float smbFootprintQuality = Filtering::ApplyBilinearFilter( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x, smbBilinearFilter );
    smbFootprintQuality = Math::Sqrt01( smbFootprintQuality );
    smbFootprintQuality *= sizeQuality; // avoid footprint momentary stretching due to changed viewing angle

    // Checkerboard resolve
    uint checkerboard = Sequence::CheckerBoard( pixelPos, gFrameIndex );
    #if( NRD_MODE == OCCLUSION )
        int3 checkerboardPos = pixelPos.xxy + int3( -1, 1, 0 );
        checkerboardPos.x = max( checkerboardPos.x, 0 );
        checkerboardPos.y = min( checkerboardPos.y, gRectSizeMinusOne.x );
        float viewZ0 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.xz ) ] );
        float viewZ1 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.yz ) ] );
        float disocclusionThresholdCheckerboard = GetDisocclusionThreshold( NRD_DISOCCLUSION_THRESHOLD, frustumSize, NoV );
        float2 wc = GetDisocclusionWeight( float2( viewZ0, viewZ1 ), viewZ, disocclusionThresholdCheckerboard );
        wc.x = ( viewZ0 > gDenoisingRange || pixelPos.x < 1 ) ? 0.0 : wc.x;
        wc.y = ( viewZ1 > gDenoisingRange || pixelPos.x >= gRectSizeMinusOne.x ) ? 0.0 : wc.y;
        wc *= Math::PositiveRcp( wc.x + wc.y );
        checkerboardPos.xy >>= 1;
    #endif

    // =====================================================================================
    // [注解] ===================== Specular 主路径 =====================
    //
    // 从这里开始，代码会并行评估两条历史候选：
    //   - SMB：表面运动历史，稳定、便宜，但对反射不总是准确
    //   - VMB：虚拟反射运动历史，更贴近镜面反射真实漂移，但也更脆弱
    //
    // 随后的核心问题不是“能不能取到历史”，而是“该信哪条历史、信多少”。
    // =====================================================================================
    // Specular

    #if( NRD_SPEC )
        // Accumulation speed
        float smbSpecHistoryConfidence = smbFootprintQuality;
        if( gHasHistoryConfidence && NRD_SUPPORTS_HISTORY_CONFIDENCE )
        {
            float confidence = saturate( gIn_SpecConfidence.SampleLevel( gLinearClamp, smbPixelUv, 0 ) );
            smbSpecHistoryConfidence = min( smbSpecHistoryConfidence, confidence );
        }
        smbSpecAccumSpeed *= lerp( smbSpecHistoryConfidence, 1.0, 1.0 / ( 1.0 + smbSpecAccumSpeed ) );

        // Current
        bool specHasData = NRD_SUPPORTS_CHECKERBOARD == 0 || gSpecCheckerboard == 2 || checkerboard == gSpecCheckerboard;
        uint2 specPos = pixelPos;
        #if( NRD_MODE == OCCLUSION )
            specPos.x >>= gSpecCheckerboard == 2 ? 0 : 1;
        #endif

        REBLUR_TYPE spec = gIn_Spec[ specPos ];

        // Checkerboard resolve // TODO: materialID support?
        #if( NRD_MODE == OCCLUSION )
            if( !specHasData )
            {
                float s0 = gIn_Spec[ checkerboardPos.xz ];
                float s1 = gIn_Spec[ checkerboardPos.yz ];

                s0 = Denanify( wc.x, s0 );
                s1 = Denanify( wc.y, s1 );

                spec = s0 * wc.x + s1 * wc.y;
            }
        #endif

        // =====================================================================================
        // [注解] 曲率估计（Curvature Estimation）
        //
        // `GetXvirtual` 需要的不只是 `hitDist`，还需要“局部表面在运动方向上弯得有多厉害”。
        // 这里就是在估计这个量：
        //   - 先根据预测运动方向构造一条局部边
        //   - 再比较边两端法线的变化量与空间跨度
        //   - 得到近似曲率 `curvature`
        //
        // 这一步并不完美，源码自己也保留了 TODO；
        // 但它足以让 specular 的 virtual reprojection 比单纯 surface motion 更贴近真实反射运动。
        // =====================================================================================
        // Curvature estimation along predicted motion ( tests 15, 40, 76, 133, 146, 147, 148 )

        /*
        TODO: curvature! (-_-)
         - by design: curvature = 0 on static objects if camera is static
         - quantization errors hurt
         - curvature on bumpy surfaces is just wrong, pulling virtual positions into a surface and introducing lags
         - suboptimal reprojection if curvature changes signs under motion
        */
        float curvature = 0.0;
        {
            // IMPORTANT: non-zero parallax on objects attached to the camera is needed
            // IMPORTANT: the direction of "deltaUv" is important ( test 1 )
            float2 uvForZeroParallax = gOrthoMode == 0.0 ? smbPixelUv : pixelUv;
            float2 deltaUv = uvForZeroParallax - Geometry::GetScreenUv( gWorldToClipPrev, Xprev + gCameraDelta.xyz ); // TODO: repeats code for "smbParallaxInPixels1" with "-" sign
            deltaUv *= gRectSize;
            deltaUv /= max( smbParallaxInPixels1, 1.0 / 256.0 );

            // 10 edge
            float3 n10, x10;
            {
                float3 xv = Geometry::ReconstructViewPosition( pixelUv + float2( 1, 0 ) * gRectSizeInv, gFrustum, 1.0, gOrthoMode );
                float3 x = Geometry::RotateVector( gViewToWorld, xv );
                float3 v = GetViewVector( x );
                float3 o = gOrthoMode == 0.0 ? 0 : x;

                x10 = o + v * dot( X - o, N ) / dot( N, v ); // line-plane intersection
                n10 = s_Normal_Roughness[ threadPos.y + NRD_BORDER ][ threadPos.x + NRD_BORDER + 1 ].xyz;
            }

            // 01 edge
            float3 n01, x01;
            {
                float3 xv = Geometry::ReconstructViewPosition( pixelUv + float2( 0, 1 ) * gRectSizeInv, gFrustum, 1.0, gOrthoMode );
                float3 x = Geometry::RotateVector( gViewToWorld, xv );
                float3 v = GetViewVector( x );
                float3 o = gOrthoMode == 0.0 ? 0 : x;

                x01 = o + v * dot( X - o, N ) / dot( N, v ); // line-plane intersection
                n01 = s_Normal_Roughness[ threadPos.y + NRD_BORDER + 1 ][ threadPos.x + NRD_BORDER ].xyz;
            }

            // Mix
            float2 w = abs( deltaUv ) + 1.0 / 256.0;
            w /= w.x + w.y; // TODO: perspective correction?

            float3 x = x10 * w.x + x01 * w.y;
            float3 n = normalize( n10 * w.x + n01 * w.y );

            // [注解] 高视差时，直接用局部法线估计曲率容易因量化误差而爆炸，
            //        所以源码会沿运动方向再探测一个更远的位置，如果仍被判定为同一表面，
            //        就用那个位置的法线/位置来替换当前估计，从而“压平”过激的曲率。
            // High parallax - flattens surface on high motion ( test 132, 172, 173, 174, 190, 201, 202, 203, e9 )

            // IMPORTANT: a must for 8-bit and 10-bit normals ( tests b7, b10, b33, 202 )
            float dither = Sequence::Bayer4x4( pixelPos, gFrameIndex ); // dithering is needed to avoid a hard-border
            float edgeFix = 1.0 - BRDF::Pow5( NoV );

            float deltaUvLenFixed = smbParallaxInPixelsMin; // "min" because not needed for objects attached to the camera!
            deltaUvLenFixed *= 1.0 + edgeFix * ( 1.0 + gFramerateScale * dither );

            float2 motionUvHigh = pixelUv + deltaUvLenFixed * deltaUv * gRectSizeInv;
            motionUvHigh = ( floor( motionUvHigh * gRectSize ) + 0.5 ) * gRectSizeInv; // Snap to the pixel center!

            if( deltaUvLenFixed > 1.0 && IsInScreenNearest( motionUvHigh ) )
            {
                float2 uvScaled = WithRectOffset( ClampUvToViewport( motionUvHigh ) );

                float zHigh = UnpackViewZ( gIn_ViewZ.SampleLevel( gNearestClamp, uvScaled, 0 ) );
                float3 xHigh = Geometry::ReconstructViewPosition( motionUvHigh, gFrustum, zHigh, gOrthoMode );
                xHigh = Geometry::RotateVector( gViewToWorld, xHigh );

                float3 nHigh = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness.SampleLevel( gNearestClamp, uvScaled, 0 ) ).xyz;

                // Replace if same surface
                float2 geometryWeightParams = GetGeometryWeightParams( NRD_CURVATURE_HIGH_PARALLAX_DISOCCLUSION_THRESHOLD, frustumSize, X, N );
                float NoX = dot( N, xHigh );

                float w = ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );
                bool cmp = w > 0.5;

                n = cmp ? nHigh : n;
                x = cmp ? xHigh : x;
            }

            // Estimate curvature for the edge { x; X }
            float3 edge = x - X;
            float edgeLenSq = Math::LengthSquared( edge );
            curvature = dot( n - N, edge ) / edgeLenSq;

            // Correction - very negative inconsistent with previous frame curvature blows up reprojection ( tests 164, 171 - 176 )
            if( curvature < 0 )
            {
                float2 uv1 = Geometry::GetScreenUv( gWorldToClipPrev, GetXvirtual( hitDistForTracking, curvature, X, X, N, V, roughness ) );
                float2 uv2 = Geometry::GetScreenUv( gWorldToClipPrev, X );
                float a = length( ( uv1 - uv2 ) * gRectSize );
                curvature *= float( a < NRD_MAX_ALLOWED_VIRTUAL_MOTION_ACCELERATION * smbParallaxInPixelsMax + gRectSizeInv.x );
            }
        }

        // =====================================================================================
        // [注解] Virtual Motion 坐标生成
        //
        // `GetXvirtual` 的实现位于 `Common.hlsli`：
        //   - 先取 GGX dominant direction
        //   - 再结合 `hitDist` 与 `curvature` 做带 magnification 的几何映射
        //   - 如果虚拟点太靠近表面，会回退地更接近 `Xprev` / `X`
        //
        // 因此这里得到的 `Xvirtual` 不是“反射射线终点”，而是一个为了时域重投影构造的
        // **虚拟反射跟踪点**。
        // =====================================================================================
        // Virtual motion - coordinates

        float3 Xvirtual = GetXvirtual( hitDistForTracking, curvature, X, Xprev, N, V, roughness );
        float XvirtualLength = length( Xvirtual );

        float2 vmbPixelUv = Geometry::GetScreenUv( gWorldToClipPrev, Xvirtual );
        vmbPixelUv = materialID == gCameraAttachedReflectionMaterialID ? smbPixelUv : vmbPixelUv;

        float2 vmbDelta = vmbPixelUv - smbPixelUv;
        float vmbPixelsTraveled = length( vmbDelta * gRectSize );

        Filtering::Bilinear vmbBilinearFilter = Filtering::GetBilinearFilter( vmbPixelUv, gRectSizePrev );
        float2 vmbBilinearGatherUv = ( vmbBilinearFilter.origin + 1.0 ) * gResourceSizeInvPrev;

        // =====================================================================================
        // [注解] VMB 置信度：粗糙度一致性
        //
        // virtual reprojection 对 roughness 非常敏感：
        //   - 上一帧 roughness 如果和当前差很多，说明 specular lobe 已显著改变
        //   - 此时复用旧的虚拟反射历史往往会造成拖影或能量错位
        //
        // 因此这里先用 roughness 生成第一层 `virtualHistoryConfidence`。
        // =====================================================================================
        // Virtual motion - confidence: roughness

        float virtualHistoryConfidence;
        float4 roughnessWeight;
        {
            float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams( roughness * roughness, gRoughnessFraction, REBLUR_ROUGHNESS_SENSITIVITY_IN_TA ); // TODO: GetRoughnessWeightParams with 0.05 sensitivity?

            #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
                float4 vmbRoughness = NRD_FrontEnd_UnpackRoughness( gPrev_Normal_Roughness.GatherBlue( gNearestClamp, vmbBilinearGatherUv ).wzxy );
            #else
                float4 vmbRoughness = NRD_FrontEnd_UnpackRoughness( gPrev_Normal_Roughness.GatherAlpha( gNearestClamp, vmbBilinearGatherUv ).wzxy );
            #endif

            roughnessWeight = ComputeNonExponentialWeightWithSigma( vmbRoughness * vmbRoughness, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y, roughnessSigma );
            roughnessWeight = lerp( Math::SmoothStep( 1.0, 0.0, smbParallaxInPixelsMax ), 1.0, roughnessWeight ); // jitter friendly

            virtualHistoryConfidence = Filtering::ApplyBilinearFilter( roughnessWeight.x, roughnessWeight.y, roughnessWeight.z, roughnessWeight.w, vmbBilinearFilter );
        }

        // [注解] 如果 SMB footprint 已经完全失效，后续某些相对法线比较就没有参考意义了；
        //        此时把 `smbNavg` 临时替换成 VMB 采样到的法线，等价于把这类比较退化成 NOP，
        //        避免“SMB 已坏但还拿它约束 VMB”这种自相矛盾的情况。
        // Patch "smbNavg" if "smb" motion is invalid ( make relative tests a NOP )

        float4 vmbNormalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.SampleLevel( STOCHASTIC_BILINEAR_FILTER, StochasticBilinear( vmbPixelUv, gRectSizePrev ) * gResolutionScalePrev, 0 ) );
        float3 vmbN = Geometry::RotateVector( gWorldPrevToWorld, vmbNormalAndRoughness.xyz );

        smbNavg = smbFootprintQuality == 0.0 ? vmbN : smbNavg;

        // =====================================================================================
        // [注解] VMB 反遮挡测试
        //
        // 和 SMB 相比，VMB 这里要同时检查：
        //   - roughness 是否还兼容
        //   - 虚拟反射点在上一帧是否仍落在相同表面附近
        //   - 法线/材质是否仍然匹配
        //
        // 只有这些条件一起成立，VMB 才会被允许参与最终历史竞争。
        // =====================================================================================
        // Virtual motion - disocclusion

        float4 vmbOcclusionWeights;
        float vmbSpecAccumSpeed;
        bool vmbAllowCatRom;
        {
            // Disocclusion: roughness
            float4 vmbOcclusion = step( 0.5, roughnessWeight );

            // Disocclusion: plane distance
            float4 vmbOcclusionThreshold = disocclusionThreshold * frustumSize;
            vmbOcclusionThreshold *= lerp( 0.25, 1.0, NoV ); // yes, "*" not "/" // TODO: it's from commit "fixed suboptimal "vmb" reprojection behavior in disocclusions", but is it really needed?
            vmbOcclusionThreshold *= float( dot( vmbN, N ) > thresholdAngle ); // good for vmb
            vmbOcclusionThreshold *= float( dot( vmbN, smbNavg ) > thresholdAngle ); // bonus check for test 168
            vmbOcclusionThreshold *= IsInScreenBilinear( vmbBilinearFilter.origin, gRectSizePrev );
            vmbOcclusionThreshold -= NRD_EPS;

            float4 vmbViewZ = UnpackViewZ( gPrev_ViewZ.GatherRed( gNearestClamp, vmbBilinearGatherUv ).wzxy );
            float3 vmbVv = Geometry::ReconstructViewPosition( vmbPixelUv, gFrustumPrev, 1.0 ); // unnormalized, orthoMode = 0
            float3 Nv = Geometry::RotateVector( gWorldToViewPrev, N );
            float NoXcurr = dot( N, Xprev - gCameraDelta.xyz );
            float4 NoXprev = ( Nv.x * vmbVv.x + Nv.y * vmbVv.y ) * ( gOrthoMode == 0 ? vmbViewZ : gOrthoMode ) + Nv.z * vmbVv.z * vmbViewZ;
            float4 vmbPlaneDist = abs( NoXprev - NoXcurr );

            vmbOcclusion *= step( vmbPlaneDist, vmbOcclusionThreshold );

            // Prev data
            uint4 vmbInternalData = gPrev_InternalData.GatherRed( gNearestClamp, vmbBilinearGatherUv ).wzxy;

            float3 vmbInternalData00 = UnpackInternalData( vmbInternalData.x );
            float3 vmbInternalData10 = UnpackInternalData( vmbInternalData.y );
            float3 vmbInternalData01 = UnpackInternalData( vmbInternalData.z );
            float3 vmbInternalData11 = UnpackInternalData( vmbInternalData.w );

            #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
                // Disocclusion: material ID
                float4 vmbMaterialID = float4( vmbInternalData00.z, vmbInternalData10.z, vmbInternalData01.z, vmbInternalData11.z  );
                vmbOcclusion *= CompareMaterials( materialID, vmbMaterialID, gSpecMinMaterial );
            #endif

            // Save disocclusion bits
            fbits += vmbOcclusion.x * 16.0;
            fbits += vmbOcclusion.y * 32.0;
            fbits += vmbOcclusion.z * 64.0;
            fbits += vmbOcclusion.w * 128.0;

            // Accumulation speed
            vmbOcclusionWeights = Filtering::GetBilinearCustomWeights( vmbBilinearFilter, vmbOcclusion );
            vmbSpecAccumSpeed = Filtering::ApplyBilinearCustomWeights( vmbInternalData00.y, vmbInternalData10.y, vmbInternalData01.y, vmbInternalData11.y, vmbOcclusionWeights );

            float vmbFootprintQuality = Filtering::ApplyBilinearFilter( vmbOcclusion.x, vmbOcclusion.y, vmbOcclusion.z, vmbOcclusion.w, vmbBilinearFilter );
            vmbFootprintQuality = Math::Sqrt01( vmbFootprintQuality );

            float vmbSpecHistoryConfidence = vmbFootprintQuality;
            if( gHasHistoryConfidence && NRD_SUPPORTS_HISTORY_CONFIDENCE )
            {
                float confidence = saturate( gIn_SpecConfidence.SampleLevel( gLinearClamp, vmbPixelUv, 0 ) );
                vmbSpecHistoryConfidence = min( vmbSpecHistoryConfidence, confidence );
            }
            vmbSpecAccumSpeed *= lerp( vmbSpecHistoryConfidence, 1.0, 1.0 / ( 1.0 + vmbSpecAccumSpeed ) );

            // Is CatRom allowed? ( requires complete "vmbOcclusion" )
            vmbAllowCatRom = dot( vmbOcclusion, 1.0 ) > 3.5 && REBLUR_USE_CATROM_FOR_VIRTUAL_MOTION_IN_TA;
            vmbAllowCatRom = vmbAllowCatRom && smbAllowCatRom; // helps to reduce over-sharpening in disoccluded areas
        }

        // =====================================================================================
        // [注解] 估计 VMB 允许的角域 / 位移范围
        //
        // 这里把“曲率造成的附加角度”和“specular lobe 自身半角”合起来，
        // 用于后面判断：上一帧与当前帧的虚拟反射点是否仍在同一可接受反射锥体中。
        //
        // 直观理解：
        //   - lobe 越宽、曲率越大、允许的虚拟位移误差就越大
        //   - lobe 越窄（镜面越锐），对 VMB 的匹配要求就越苛刻
        // =====================================================================================
        // Estimate how many pixels are traveled by virtual motion - how many radians can it be?

        float tanHalfAngle;
        float curvatureAngle;
        float lobeHalfAngle;
        {
            // IMPORTANT: if curvature angle is multiplied by path length then we can get an angle exceeding "2 * PI", what is impossible.
            // The max angle is PI ( most left and most right points on a hemisphere ), it can be achieved by using "tan" instead of angle.
            float curvatureAngleTan = pixelSize * abs( curvature ); // tana = pixelSize / curvatureRadius = pixelSize * curvature
            curvatureAngleTan *= max( vmbPixelsTraveled / max( NoV, 0.01 ), 1.0 ); // path length
            curvatureAngleTan *= 2.0; // TODO: why it's here? but works well

            curvatureAngle = atan( curvatureAngleTan );

            // Copied from "GetNormalWeightParam" but doesn't use "lobeAngleFraction"
            float percentOfVolume = NRD_MAX_PERCENT_OF_LOBE_VOLUME / ( 1.0 + vmbSpecAccumSpeed );
            float lobeTanHalfAngle = ImportanceSampling::GetSpecularLobeTanHalfAngle( roughnessModified, percentOfVolume );

            // TODO: use old code and sync with "GetNormalWeightParam"?
            //float lobeTanHalfAngle = ImportanceSampling::GetSpecularLobeTanHalfAngle( roughnessModified, NRD_MAX_PERCENT_OF_LOBE_VOLUME );
            //lobeTanHalfAngle /= 1.0 + vmbSpecAccumSpeed;

            lobeHalfAngle = max( atan( lobeTanHalfAngle ), NRD_NORMAL_ENCODING_ERROR );

            tanHalfAngle = lobeTanHalfAngle + curvatureAngleTan;
        }

        // =====================================================================================
        // [注解] VMB 置信度：视差一致性
        //
        // 这里会拿“当前虚拟点”和“上一帧 tracking hit distance 反推的虚拟点”做比较，
        // 如果两者在屏幕上的偏差超过 specular lobe 能容忍的范围，`parallaxWeight` 就会下降。
        // =====================================================================================
        // Virtual motion - confidence: parallax

        // Tests 3, 6, 8, 11, 14, 100, 103, 104, 106, 109, 110, 114, 120, 127, 130, 131, 132, 138, 139 and 9e
        float parallaxWeight;
        {
            float hitDistForTrackingPrev = gPrev_SpecHitDistForTracking.SampleLevel( gLinearClamp, vmbPixelUv * gResolutionScalePrev, 0 );
            float3 XvirtualPrev = GetXvirtual( hitDistForTrackingPrev, curvature, X, Xprev, N, V, roughness );

            float2 vmbPixelUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, XvirtualPrev );
            vmbPixelUvPrev = materialID == gCameraAttachedReflectionMaterialID ? smbPixelUv : vmbPixelUvPrev;

            float pixelSizeAtXvirtual = PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, XvirtualLength );
            float r = tanHalfAngle * min( hitDistForTracking, hitDistForTrackingPrev ) / pixelSizeAtXvirtual; // "pixelSize" at "XvirtualPrev" seems to be not needed
            float d = length( ( vmbPixelUvPrev - vmbPixelUv ) * gRectSize );

            r = max( r, 0.1 ); // important, especially if "curvatureAngle" is not used

            parallaxWeight = Math::LinearStep( r, 0.0, d ); // "r" can be scaled down to strengthen the test
        }

        // [注解] 第二层是法线一致性。
        //        注意这里调用的是 `GetEncodingAwareNormalWeight`（定义在 `Common.hlsli`），
        //        它会把编码误差阈值、曲率角补偿一起考虑进去，
        //        比直接比较 dot(Ncurr, Nprev) 更贴近真实可接受范围。
        // Virtual motion - confidence: normal

        {
            float normalWeight = GetEncodingAwareNormalWeight( N, vmbN, lobeHalfAngle, curvatureAngle, REBLUR_NORMAL_ULP );
            normalWeight = lerp( Math::SmoothStep( 1.0, 0.0, vmbPixelsTraveled ), 1.0, normalWeight ); // jitter friendly

            virtualHistoryConfidence *= normalWeight;
        }

        // =====================================================================================
        // [注解] VMB 置信度：prev-prev 连续性检查
        //
        // 仅靠“当前帧 vs 上一帧”匹配还不够，VMB 还会再沿着虚拟运动方向往回看一小步：
        //   - 如果前一帧再往前的法线/roughness 也连续，说明这条虚拟历史更可信
        //   - 如果只在单帧上偶然匹配，但时间上不连续，就会在这里被压低
        // =====================================================================================
        // Virtual motion - confidence: prev-prev tests

        {
            // IMPORTANT: 2 is needed because:
            // - line *** allows fallback to laggy surface motion, which can be wrongly redistributed by virtual motion
            // - we use at least linear filters, as the result a wider initial offset is needed
            float stepBetweenTaps = min( vmbPixelsTraveled * gFramerateScale, 2.0 ) + vmbPixelsTraveled / REBLUR_VIRTUAL_MOTION_PREV_PREV_WEIGHT_ITERATION_NUM;
            vmbDelta *= Math::Rsqrt( Math::LengthSquared( vmbDelta ) );
            vmbDelta /= gRectSizePrev;

            float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams( vmbNormalAndRoughness.w * vmbNormalAndRoughness.w, gRoughnessFraction, REBLUR_ROUGHNESS_SENSITIVITY_IN_TA ); // TODO: GetRoughnessWeightParams with 0.05 sensitivity?

            [unroll]
            for( i = 1; i <= REBLUR_VIRTUAL_MOTION_PREV_PREV_WEIGHT_ITERATION_NUM; i++ )
            {
                float2 vmbPixelUvPrev = vmbPixelUv + vmbDelta * i * stepBetweenTaps;
                float4 vmbNormalAndRoughnessPrev = NRD_FrontEnd_UnpackNormalAndRoughness( gPrev_Normal_Roughness.SampleLevel( STOCHASTIC_BILINEAR_FILTER, StochasticBilinear( vmbPixelUvPrev, gRectSizePrev ) * gResolutionScalePrev, 0 ) );

                float w = GetEncodingAwareNormalWeight( vmbNormalAndRoughness.xyz, vmbNormalAndRoughnessPrev.xyz, lobeHalfAngle, curvatureAngle * ( 1.0 + i * stepBetweenTaps ), REBLUR_NORMAL_ULP );
                w *= ComputeNonExponentialWeightWithSigma( vmbNormalAndRoughnessPrev.w * vmbNormalAndRoughnessPrev.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y, roughnessSigma );

                #if( REBLUR_USE_STF == 1 && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
                    // Cures issues of "StochasticBilinear" and produces closer look to the linear filter
                    w = lerp( 1.0, w, saturate( stepBetweenTaps ) );
                #endif

                w = IsInScreenNearest( vmbPixelUvPrev ) ? w : 1.0;

                // For "min" usage "virtualHistoryConfidence" must include only "roughness" and "normal" weights before this line
                virtualHistoryConfidence = min( virtualHistoryConfidence, w );
            }
        }

        // Virtual motion - confidence: apply parallax weight
        virtualHistoryConfidence *= parallaxWeight;

        // =====================================================================================
        // [注解] Surface history confidence
        //
        // 这是给 SMB 单独算的一份置信度：
        //   - 当 VMB 因凹凸表面、曲率过大、法线跳变而不靠谱时，SMB 仍可能更稳
        //   - 因此源码不会盲目偏向 VMB，而是同时维护两条历史各自能积累多少帧
        // =====================================================================================
        // Surface history confidence ( test 9, 9e )

        // IMPORTANT: needs to be responsive, because "vmb" fails on bumpy surfaces for the following reasons:
        //  - normal and prev-prev tests fail
        //  - curvature is so high that "vmb" regresses to "smb" and starts to lag
        float surfaceHistoryConfidence;
        {
            float a = atan( smbParallaxInPixelsMax * pixelSize / length( X ) );
            //a = acos( saturate( dot( V, smbVprev ) ) ); // numerically unstable

            float nonLinearAccumSpeed = 1.0 / ( 1.0 + smbSpecAccumSpeed );
            float hPrev = ExtractHitDist( gHistory_Spec.SampleLevel( gLinearClamp, smbPixelUv * gResolutionScalePrev, 0 ) ); // this is safe because "history" is always "cleared" on startup, the rest is handled by "lerp" below
            float h = lerp( hPrev, ExtractHitDist( spec ), nonLinearAccumSpeed ) * hitDistNormalization;

            float tana0 = ImportanceSampling::GetSpecularLobeTanHalfAngle( roughnessModified, NRD_MAX_PERCENT_OF_LOBE_VOLUME ); // base lobe angle
            tana0 *= lerp( NoV, 1.0, roughnessModified ); // make more strict if NoV is low and lobe is very V-dependent
            tana0 *= nonLinearAccumSpeed; // make more strict if history is long
            tana0 /= GetHitDistFactor( h, frustumSize ) + NRD_EPS; // make relaxed "in corners", where reflection is close to the surface

            float a0 = max( atan( tana0 ), NRD_NORMAL_ENCODING_ERROR );

            float f = Math::LinearStep( a0, 0.0, a );
            surfaceHistoryConfidence = Math::Pow01( f, 4.0 );

            // Lerp to "1" for very high roughness, where specular motion regresses to surface motion ( test 236 )
            f = Math::LinearStep( 0.8, 0.9, roughnessModified );
            surfaceHistoryConfidence = lerp( surfaceHistoryConfidence, 1.0, f );
        }

        // Limit number of accumulated frames
        float smbSpecAccumSpeed_NoHistoryFix;
        float vmbSpecAccumSpeed_NoHistoryFix;
        {
            // Responsive accumulation
            float responsiveFactor = RemapRoughnessToResponsiveFactor( roughness );
            float smc = GetSpecMagicCurve( roughnessModified );

            float2 f;
            f.x = dot( N, normalize( smbNavg ) );
            f.y = dot( N, vmbN );
            f = lerp( smc, 1.0, responsiveFactor ) * Math::Pow01( f, lerp( 32.0, 1.0, smc ) * ( 1.0 - responsiveFactor ) );

            float2 maxResponsiveFrameNum = gMaxAccumulatedFrameNum;
            maxResponsiveFrameNum *= f;
            maxResponsiveFrameNum = max( maxResponsiveFrameNum, gResponsiveAccumulationMinAccumulatedFrameNum );

            // Apply limits
            float2 maxFrameNum = gMaxAccumulatedFrameNum * float2( surfaceHistoryConfidence, virtualHistoryConfidence );
            float2 maxFrameNum_NoHistoryFix = min( maxFrameNum, max( maxResponsiveFrameNum, gHistoryFixFrameNum ) );

            smbSpecAccumSpeed_NoHistoryFix = min( smbSpecAccumSpeed, maxFrameNum_NoHistoryFix.x );
            vmbSpecAccumSpeed_NoHistoryFix = min( vmbSpecAccumSpeed, maxFrameNum_NoHistoryFix.y );

            maxFrameNum = min( maxFrameNum, maxResponsiveFrameNum );

            smbSpecAccumSpeed = min( smbSpecAccumSpeed, maxFrameNum.x );
            vmbSpecAccumSpeed = min( vmbSpecAccumSpeed, maxFrameNum.y );
        }

        // =====================================================================================
        // [注解] 选择 SMB 还是 VMB
        //
        // 虽然变量名叫 `virtualHistoryAmount`，但这里最终会用 `step(0.5, ...)` 把它二值化：
        //   - 0：选 SMB
        //   - 1：选 VMB
        //
        // 也就是说，这里更像“赢家通吃”的历史路径选择器，而不是连续插值权重。
        // 这么做能减少两条不完全一致的历史在边界区域互相拉扯。
        // =====================================================================================
        // Virtual history amount ( tests 65, 66, 103, 107, 111, 132, e9, e11, 218 ) // ***

        // OLD: virtualHistoryAmount = saturate( scale )
        //      * Dfactor                   - 1 is assumed now, because "Dfactor" is applied in "GetXvirtual" to "vmbPixelUv" making it closer to surface where needed ( test 236 )
        //    Helped on bumpy surfaces, because virtual motion got ruined by big curvature
        //      * normalBasedConfidence     - 1 is assumed now, because the selector below does the same and avoids "double applying" ( was used before "prev-prev" test )
        //    Helped to preserve "lying-on-surface" roughness details
        //      * roughnessBasedConfidence  - 1 is assumed now, because the selector below does the same and avoids "double applying"
        float virtualHistoryAmount;
        {
            // "1" if "vmb" >= "smb", pull towards "smb" based on delta otherwise
            virtualHistoryAmount = 1.0 + ( vmbSpecAccumSpeed - smbSpecAccumSpeed ) / ( 1.0 + 0.5 * max( vmbSpecAccumSpeed, smbSpecAccumSpeed ) ); // TODO: 0.5 => 0.25?
            virtualHistoryAmount = saturate( virtualHistoryAmount );

            // Choose one: "smb" or "vmb"
            virtualHistoryAmount = step( 0.5, virtualHistoryAmount ); // TODO: dithering seems to be not needed, since "vmb" is dominating for any possible "roughness, NoV"
        }

        // =====================================================================================
        // [注解] 采样 specular 历史
        //
        // `BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights` 定义在 `REBLUR_Common.hlsli`：
        //   - 主 history（`gHistory_Spec`）尽量走更锐利的 CatRom
        //   - fast history / SH history 走更稳妥的 bilinear with custom weights
        //   - 如果 footprint 不完整，则会自动回退到 bilinear
        //
        // 这也是 `smbAllowCatRom` / `vmbAllowCatRom` 必须提前算好的原因。
        // =====================================================================================
        // Sample history
        REBLUR_TYPE specHistory;

        REBLUR_FAST_TYPE specFastHistory;
        REBLUR_SH_TYPE specShHistory;
        {
            float2 uv = lerp( smbPixelUv, vmbPixelUv, virtualHistoryAmount );
            float4 occlusionWeights = lerp( smbOcclusionWeights, vmbOcclusionWeights, virtualHistoryAmount );
            bool allowCatRom = virtualHistoryAmount < 0.5 ? smbAllowCatRom : vmbAllowCatRom;

            BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
                saturate( uv ) * gRectSizePrev, gResourceSizeInvPrev,
                occlusionWeights, allowCatRom,
                gHistory_Spec, specHistory,
                gHistory_SpecFast, specFastHistory
                #if( NRD_MODE == SH )
                    , gHistory_SpecSh, specShHistory
                #endif
            );

            // Avoid negative values
            specHistory = ClampNegativeToZero( specHistory );
        }

        // =====================================================================================
        // [注解] Specular 历史融合
        //
        // `specAccumSpeed` 来自 SMB/VMB 两条路径二选一后的历史长度；
        // `specNonLinearAccumSpeed = 1 / (1 + specAccumSpeed)` 是实际混合当前帧的权重。
        //
        // `MixHistoryAndCurrent` 在 `REBLUR_Common.hlsli` 中定义：
        //   - 颜色分量正常线性插值
        //   - hit distance 分量会被额外限制最小混合速度，避免长期历史导致 hit distance 过度僵化
        // =====================================================================================
        // Accumulation
        float specAccumSpeedCorrected = lerp( smbSpecAccumSpeed_NoHistoryFix, vmbSpecAccumSpeed_NoHistoryFix, virtualHistoryAmount ); // avoid "HistoryFix" in responsive accumulation

        float specAccumSpeed = lerp( smbSpecAccumSpeed, vmbSpecAccumSpeed, virtualHistoryAmount );
        float specNonLinearAccumSpeed = 1.0 / ( 1.0 + specAccumSpeed );

        if( !specHasData )
            specNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, specNonLinearAccumSpeed );

        REBLUR_TYPE specResult = MixHistoryAndCurrent( specHistory, spec, specNonLinearAccumSpeed, roughnessModified );

        #if( NRD_MODE == SH )
            REBLUR_SH_TYPE specSh = gIn_SpecSh[ specPos ];
            REBLUR_SH_TYPE specShResult = lerp( specShHistory, specSh, specNonLinearAccumSpeed );
        #endif

        // =====================================================================================
        // [注解] Firefly Suppressor（抑制稀疏高亮闪烁）
        //
        // 随着历史变长，允许的相对亮度峰值会逐步收紧：
        //   - 先约束辐亮度 / luma，抑制孤立高亮点被历史放大
        //   - 再约束 `.w` 中的 hit distance，避免后续基于 hit distance 的权重判断被异常值污染
        //
        // 这里使用的阈值来自 `REBLUR_Config.hlsli`：
        //   `REBLUR_FIREFLY_SUPPRESSOR_MAX_RELATIVE_INTENSITY`
        //   `REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE`
        //   `REBLUR_FIREFLY_SUPPRESSOR_FAST_RELATIVE_INTENSITY`
        // =====================================================================================
        // Firefly suppressor
        float specMaxRelativeIntensity = gFireflySuppressorMinRelativeScale + REBLUR_FIREFLY_SUPPRESSOR_MAX_RELATIVE_INTENSITY / ( specAccumSpeed + 1.0 );


        float specAntifireflyFactor = specAccumSpeed * gMaxBlurRadius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE;
        specAntifireflyFactor /= 1.0 + specAntifireflyFactor;

        #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
        {
            float specLumaResult = GetLuma( specResult );
            float specLumaClamped = min( specLumaResult, GetLuma( specHistory ) * specMaxRelativeIntensity );
            specLumaClamped = lerp( specLumaResult, specLumaClamped, specAntifireflyFactor );

            specResult = ChangeLuma( specResult, specLumaClamped );
            #if( NRD_MODE == SH )
                specShResult.xyz *= GetLumaScale( length( specShResult.xyz ), specLumaClamped );
            #endif

            // This is required for "hit distance weight" to work
            float specHitDistMaxRelativeIntensity = 1.2 + 1.0 / ( specAccumSpeed + 1.0 );
            specResult.w = lerp( specResult.w, min( specResult.w, specHistory.w * specHitDistMaxRelativeIntensity ), specAntifireflyFactor );
        }
        #endif

        // Output
        gOut_Spec[ pixelPos ] = specResult;
        #if( NRD_MODE == SH )
            gOut_SpecSh[ pixelPos ] = specShResult;
        #endif

        // [注解] Fast history 是给后续更敏捷的响应式控制使用的亮度历史，
        //        它积累得更短、更快，常用于辅助抑制 crawling 与快速变化区域。
        // Fast history
        {

            float maxFastAccumulatedFrameNum = gMaxFastAccumulatedFrameNum;
            if( materialID == gStrandMaterialID )
                maxFastAccumulatedFrameNum = max( maxFastAccumulatedFrameNum, gMaxAccumulatedFrameNum / 5 );

            float specHistoryConfidence = lerp( surfaceHistoryConfidence, virtualHistoryConfidence, virtualHistoryAmount );
            float specFastNonLinearAccumSpeed = GetNonLinearAccumSpeed( specAccumSpeed, maxFastAccumulatedFrameNum, specHistoryConfidence, specHasData );
            float specFastResult = lerp( specFastHistory, GetLuma( spec ), specFastNonLinearAccumSpeed );

            // Firefly suppressor ( fixes heavy crawling under camera rotation: test 95, 120 )
            #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
                float specFastClamped = min( specFastResult, GetLuma( specHistory ) * specMaxRelativeIntensity * REBLUR_FIREFLY_SUPPRESSOR_FAST_RELATIVE_INTENSITY );
                specFastResult = lerp( specFastResult, specFastClamped, specAntifireflyFactor );
            #endif

            gOut_SpecFast[ pixelPos ] = specFastResult;
        }

        // [注解] Debug 模式下会复用 `virtualHistoryAmount` 这个通道输出不同可视化量，
        //        例如曲率符号、surface/VMB 置信度、tracking hit distance 等。
        //        这也是为什么它后面会被重新赋值，而不是单纯保留 0/1 选择结果。
        // Debug

        #if( REBLUR_SHOW == REBLUR_SHOW_CURVATURE )
            virtualHistoryAmount = abs( curvature ) * pixelSize * 30.0;
        #elif( REBLUR_SHOW == REBLUR_SHOW_CURVATURE_SIGN )
            virtualHistoryAmount = sign( curvature ) * 0.5 + 0.5;
        #elif( REBLUR_SHOW == REBLUR_SHOW_SURFACE_HISTORY_CONFIDENCE )
            virtualHistoryAmount = surfaceHistoryConfidence;
        #elif( REBLUR_SHOW == REBLUR_SHOW_VIRTUAL_HISTORY_CONFIDENCE )
            virtualHistoryAmount = virtualHistoryConfidence;
        #elif( REBLUR_SHOW == REBLUR_SHOW_HIT_DIST_FOR_TRACKING )
            float smc = GetSpecMagicCurve( roughness );
            virtualHistoryAmount = hitDistForTracking * lerp( 1.0, 5.0, smc ) / ( 1.0 + hitDistForTracking * lerp( 1.0, 5.0, smc ) );
        #endif
    #else
        float specAccumSpeedCorrected = 0;
        float curvature = 0;
        float virtualHistoryAmount = 0;
    #endif

    // =====================================================================================
    // [注解] 输出 `Data2`
    //
    // `PackData2` 定义在 `REBLUR_Common.hlsli`，会把下面几类信息塞进一个 `uint`：
    //   - SMB / VMB 的反遮挡 bitmask
    //   - `virtualHistoryAmount`
    //   - `curvature`
    //   - `smbAllowCatRom`
    //
    // 这些值会被后续 `TemporalStabilization` 等 pass 解包继续使用，
    // 所以它不是调试残留，而是正式管线状态的一部分。
    // =====================================================================================
    // Output
    #if( NRD_MODE != OCCLUSION )
        // TODO: "PackData2" can be inlined into the code ( right after a variable gets ready for use ) to utilize the only
        // one "uint" for the intermediate storage. But it looks like the compiler does good job by rearranging the code for us
        gOut_Data2[ pixelPos ] = PackData2( fbits, curvature, virtualHistoryAmount, smbAllowCatRom );
    #endif

    // =====================================================================================
    // [注解] ===================== Diffuse 路径 =====================
    //
    // Diffuse 明显简单很多：
    //   - 只有 SMB，没有 VMB
    //   - 没有曲率、虚拟反射点、双路径竞争
    //   - 但仍然保留 checkerboard resolve、history confidence、fast history、firefly suppressor
    //
    // 也正因为如此，specular 的复杂度主要都集中在上面那一大段。
    // =====================================================================================
    // Diffuse

    #if( NRD_DIFF )
        // Accumulation speed
        float diffHistoryConfidence = smbFootprintQuality;
        if( gHasHistoryConfidence && NRD_SUPPORTS_HISTORY_CONFIDENCE )
        {
            float confidence = saturate( gIn_DiffConfidence.SampleLevel( gLinearClamp, smbPixelUv, 0 ) );
            diffHistoryConfidence = min( diffHistoryConfidence, confidence );
        }
        diffAccumSpeed *= lerp( diffHistoryConfidence, 1.0, 1.0 / ( 1.0 + diffAccumSpeed ) );

        // Current
        bool diffHasData = NRD_SUPPORTS_CHECKERBOARD == 0 || gDiffCheckerboard == 2 || checkerboard == gDiffCheckerboard;
        uint2 diffPos = pixelPos;
        #if( NRD_MODE == OCCLUSION )
            diffPos.x >>= gDiffCheckerboard == 2 ? 0 : 1;
        #endif

        REBLUR_TYPE diff = gIn_Diff[ diffPos ];

        // Checkerboard resolve // TODO: materialID support?
        #if( NRD_MODE == OCCLUSION )
            if( !diffHasData )
            {
                float d0 = gIn_Diff[ checkerboardPos.xz ];
                float d1 = gIn_Diff[ checkerboardPos.yz ];

                d0 = Denanify( wc.x, d0 );
                d1 = Denanify( wc.y, d1 );

                diff = d0 * wc.x + d1 * wc.y;
            }
        #endif

        // [注解] Diffuse 的历史采样完全沿用 SMB footprint，
        //        不需要像 specular 那样在 SMB/VMB 之间竞争，因此逻辑显著更直接。
        // Sample history
        REBLUR_TYPE diffHistory;

        REBLUR_FAST_TYPE diffFastHistory;
        REBLUR_SH_TYPE diffShHistory;

        BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
            saturate( smbPixelUv ) * gRectSizePrev, gResourceSizeInvPrev,
            smbOcclusionWeights, smbAllowCatRom,
            gHistory_Diff, diffHistory,
            gHistory_DiffFast, diffFastHistory
            #if( NRD_MODE == SH )
                , gHistory_DiffSh, diffShHistory
            #endif
        );

        // Avoid negative values
        diffHistory = ClampNegativeToZero( diffHistory );

        // [注解] Diffuse 这里直接使用 SMB 恢复出的 `diffAccumSpeed`，
        //        因为没有 VMB，所以也不存在 `virtualHistoryAmount` 选择器。
        // Accumulation
        float diffNonLinearAccumSpeed = 1.0 / ( 1.0 + diffAccumSpeed );


        if( !diffHasData )
            diffNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, diffNonLinearAccumSpeed );

        REBLUR_TYPE diffResult = MixHistoryAndCurrent( diffHistory, diff, diffNonLinearAccumSpeed );
        #if( NRD_MODE == SH )
            REBLUR_SH_TYPE diffSh = gIn_DiffSh[ diffPos ];
            REBLUR_SH_TYPE diffShResult = lerp( diffShHistory, diffSh, diffNonLinearAccumSpeed );
        #endif

        // [注解] Diffuse 同样会做 firefly 抑制，但少了 specular 那套 VMB 相关不稳定性，
        //        因此这一段更多是在压制稀疏高亮和相机转动时的爬行噪点。
        // Firefly suppressor
        #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )

            float diffMaxRelativeIntensity = gFireflySuppressorMinRelativeScale + REBLUR_FIREFLY_SUPPRESSOR_MAX_RELATIVE_INTENSITY / ( diffAccumSpeed + 1.0 );

            float diffAntifireflyFactor = diffAccumSpeed * gMaxBlurRadius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE;
            diffAntifireflyFactor /= 1.0 + diffAntifireflyFactor;

            float diffLumaResult = GetLuma( diffResult );
            float diffLumaClamped = min( diffLumaResult, GetLuma( diffHistory ) * diffMaxRelativeIntensity );
            diffLumaClamped = lerp( diffLumaResult, diffLumaClamped, diffAntifireflyFactor );

            diffResult = ChangeLuma( diffResult, diffLumaClamped );
            #if( NRD_MODE == SH )
                diffShResult.xyz *= GetLumaScale( length( diffShResult.xyz ), diffLumaClamped );
            #endif

            // This is required for "hit distance weight" to work
            float diffHitDistMaxRelativeIntensity = 1.2 + 1.0 / ( diffAccumSpeed + 1.0 );
            diffResult.w = lerp( diffResult.w, min( diffResult.w, diffHistory.w * diffHitDistMaxRelativeIntensity ), diffAntifireflyFactor );
        #endif

        // Output
        gOut_Diff[ pixelPos ] = diffResult;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ pixelPos ] = diffShResult;
        #endif

        // Fast history
        float diffFastAccumSpeed = min( diffAccumSpeed, gMaxFastAccumulatedFrameNum );
        float diffFastNonLinearAccumSpeed = 1.0 / ( 1.0 + diffFastAccumSpeed );

        if( !diffHasData )
            diffFastNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, diffFastNonLinearAccumSpeed );

        float diffFastResult = lerp( diffFastHistory, GetLuma( diff ), diffFastNonLinearAccumSpeed );

        #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
            // Firefly suppressor ( fixes heavy crawling under camera rotation, test 99 )
            float diffFastClamped = min( diffFastResult, GetLuma( diffHistory ) * diffMaxRelativeIntensity * REBLUR_FIREFLY_SUPPRESSOR_FAST_RELATIVE_INTENSITY );
            diffFastResult = lerp( diffFastResult, diffFastClamped, diffAntifireflyFactor );
        #endif

        gOut_DiffFast[ pixelPos ] = diffFastResult;
    #else
        float diffAccumSpeed = 0;
    #endif

    // =====================================================================================
    // [注解] 输出 `Data1`
    //
    // `PackData1` 会把当前帧完成 temporal accumulation 后得到的
    // diffuse / specular 累积帧数写成紧凑格式，供后续：
    //   - `HistoryFix`
    //   - `Blur`
    //   - `PostBlur`
    //
    // 尤其是空间滤波阶段会根据这个值推导 non-linear accumulation speed，
    // 决定“历史越稳定时，空间模糊是否可以更保守”。
    // =====================================================================================
    // Output
    gOut_Data1[ pixelPos ] = PackData1( diffAccumSpeed, specAccumSpeedCorrected );

}
