/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR Pre-Pass（预处理 pass）
//
// 这是 REBLUR 降噪管线的 **第二个 pass**（紧接 HitDistReconstruction 之后）。
// 主要职责：
//   1. **棋盘格采样的恢复（Checkerboard Resolve）**：
//      如果使用了半分辨率棋盘格渲染（隔一个像素采样一次来节省光追开销），
//      此 pass 负责通过邻域插值将缺失像素填充回来
//   2. **空间预滤波（Spatial Pre-Filter）**：
//      在正式的时空降噪之前，做一次轻量级的空间模糊
//      - 对 Diffuse 和 Specular 分别执行（通过 #include 同一个 SpatialFilter 文件两次）
//      - Specular 的模糊范围受 hit distance 和 specular lobe 限制（"in-lobe" 模糊）
//   3. **Specular Hit Distance for Tracking**：
//      计算用于运动估计的最小 hit distance（仅 specular），用于后续的 temporal reprojection
//
// 此 pass 使用 Poisson 采样盘进行空间滤波，而非 HitDistReconstruction 中的规则邻域。
// =====================================================================================

// [注解] NRD 核心头文件
#include "NRD.hlsli"

// [注解] 数学库
#include "ml.hlsli"

// [注解] REBLUR 配置参数
//        关键宏：REBLUR_PRE_PASS, REBLUR_PRE_PASS_ROTATOR_MODE,
//               REBLUR_PRE_PASS_POISSON_SAMPLE_NUM, REBLUR_PRE_PASS_RADIUS_SCALE,
//               REBLUR_MAX_PERCENT_OF_LOBE_VOLUME_FOR_PRE_PASS 等
#include "REBLUR_Config.hlsli"

// [注解] 自动生成的资源绑定声明
//        输入：gIn_Tiles, gIn_ViewZ, gIn_Normal_Roughness, gIn_Diff, gIn_Spec, gIn_DiffSh, gIn_SpecSh
//        输出：gOut_Diff, gOut_Spec, gOut_DiffSh, gOut_SpecSh, gOut_SpecHitDistForTracking
//        以及各种常量 buffer 参数
#include "REBLUR_PrePass.resources.hlsli"

// [注解] 通用工具函数
#include "Common.hlsli"

// [注解] REBLUR 公共工具函数
#include "REBLUR_Common.hlsli"

// =====================================================================================
// [注解] 主 Compute Shader 入口
//
// 注意此 pass **没有 shared memory / preload**（与 HitDistReconstruction 不同）
// 因为 Poisson 采样点是分散的（不是规则邻域），无法有效利用 shared memory
// =====================================================================================
[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    // [注解] NRD_CTA_ORDER_REVERSED: 与 NRD_CTA_ORDER_DEFAULT 类似，但使用反向排列
    //        这可以改善某些情况下的 cache coherence
    //        展开后得到 pixelPos 和 threadPos
    NRD_CTA_ORDER_REVERSED;

    // =====================================================================================
    // [注解] Tile 级别 early out（与 HitDistReconstruction 相同）
    // =====================================================================================
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // [注解] 像素级 early out：深度超出降噪范围
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pixelPos ) ] );
    if( viewZ > gDenoisingRange )
        return;

    // =====================================================================================
    // [注解] 中心像素数据准备
    // =====================================================================================

    // [注解] 解包法线 + 粗糙度 + 材质 ID
    //        materialID: REBLUR 使用材质 ID 来区分不同材质的像素
    //        不同材质的像素不应互相模糊（如金属 vs 塑料），这是 REBLUR 比 RELAX 更细致的地方
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ], materialID );
    float3 N = normalAndRoughness.xyz;       // [注解] 世界空间法线
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );  // [注解] 观察空间法线
    float roughness = normalAndRoughness.w;  // [注解] 粗糙度

    // [注解] 像素 UV（归一化坐标，像素中心）
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;

    // [注解] 从 UV + viewZ 重建观察空间 3D 位置
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );

    // [注解] 观察方向（归一化，从表面指向相机）
    float3 Vv = GetViewVector( Xv, true );

    // [注解] NoV = |dot(法线, 视线方向)|：法线与视线的夹角余弦的绝对值
    //        用于多种权重计算（掠射角修正等）
    float NoV = abs( dot( Nv, Vv ) );

    // [注解] frustumSize: 在当前深度下一个像素对应的世界空间大小
    const float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );

    // [注解] 获取模糊核的旋转矩阵
    //        REBLUR_PRE_PASS_ROTATOR_MODE: 旋转模式（随机旋转/基于帧的旋转等）
    //        gRotatorPre: 预设旋转参数
    //        gFrameIndex: 当前帧索引（用于时序旋转）
    //        旋转 Poisson 采样盘可以减少结构化 artifact（摩尔纹等）
    const float4 rotator = GetBlurKernelRotation( REBLUR_PRE_PASS_ROTATOR_MODE, pixelPos, gRotatorPre, gFrameIndex );

    // =====================================================================================
    // [注解] 棋盘格采样恢复（Checkerboard Resolve）
    //
    // 背景：为了节省光追开销，可以使用 "棋盘格渲染"：
    //   偶数帧只在偶数位置的像素做光追，奇数帧只在奇数位置做
    //   这样每帧只需一半的光追射线
    //
    // 这里预计算棋盘格恢复所需的权重 wc：
    //   对于当前帧未采样的像素，用左右两个有效邻居的加权平均来填充
    //   权重基于深度一致性（disocclusion threshold）
    //   - viewZ0: 左邻居的深度
    //   - viewZ1: 右邻居的深度
    //   - 如果邻居深度与中心深度差异过大（不连续），权重设为 0
    //   - 如果邻居在屏幕外或超出降噪范围，权重也设为 0
    //
    // NRD_SUPPORTS_CHECKERBOARD == 1 时才编译此段
    // =====================================================================================
#if( NRD_SUPPORTS_CHECKERBOARD == 1 )
    // [注解] 判断当前像素在棋盘格中的奇偶性
    uint checkerboard = Sequence::CheckerBoard( pixelPos, gFrameIndex );

    // [注解] 计算左右邻居的位置
    //        checkerboardPos.x = 左邻居 x 坐标（至少为 0）
    //        checkerboardPos.y = 右邻居 x 坐标（至多为 gRectSizeMinusOne.x）
    //        checkerboardPos.z = 当前像素 y 坐标（左右邻居 y 相同）
    int3 checkerboardPos = pixelPos.xxy + int3( -1, 1, 0 );
    checkerboardPos.x = max( checkerboardPos.x, 0 );
    checkerboardPos.y = min( checkerboardPos.y, gRectSizeMinusOne.x );

    // [注解] 读取左右邻居的深度
    float viewZ0 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.xz ) ] );
    float viewZ1 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.yz ) ] );

    // [注解] 计算 disocclusion 阈值：基于 frustumSize 和 NoV 调整
    //        掠射角（NoV 小）时阈值更宽松
    float disocclusionThresholdCheckerboard = GetDisocclusionThreshold( NRD_DISOCCLUSION_THRESHOLD, frustumSize, NoV );

    // [注解] 计算左右邻居的权重（基于深度一致性）
    float2 wc = GetDisocclusionWeight( float2( viewZ0, viewZ1 ), viewZ, disocclusionThresholdCheckerboard );

    // [注解] 边界/有效性检查
    wc.x = ( viewZ0 > gDenoisingRange || pixelPos.x < 1 ) ? 0.0 : wc.x;
    wc.y = ( viewZ1 > gDenoisingRange || pixelPos.x >= gRectSizeMinusOne.x ) ? 0.0 : wc.y;

    // [注解] 归一化权重使其和为 1（Math::PositiveRcp 安全除法，防止除以 0）
    wc *= Math::PositiveRcp( wc.x + wc.y );

    // [注解] 棋盘格模式下，输入纹理是半宽的（水平方向只有一半像素）
    //        所以 x 坐标需要右移 1 位（除以 2）来索引输入纹理
    checkerboardPos.xy >>= 1;
#endif

    // =====================================================================================
    // [注解] 非线性累积速度（Non-linear Accumulation Speed）
    //
    // 这是一个 float2(diffSpeed, specSpeed)
    // 在 Pre-Pass 中，这个值是常量 REBLUR_PRE_PASS_NON_LINEAR_ACCUM_SPEED
    // 在 Blur/PostBlur pass 中，这个值来自 temporal accumulation 的输出
    //   值越小 → 说明累积了更多历史帧 → blur radius 可以更小
    //   值越大 → 说明是新出现的区域 → blur radius 需要更大
    //
    // Pre-Pass 中使用固定值（因为还没有进行 temporal accumulation）
    // =====================================================================================
    float2 nonLinearAccumSpeed = REBLUR_PRE_PASS_NON_LINEAR_ACCUM_SPEED;

    // =====================================================================================
    // [注解] 空间滤波核心 —— 通过 #include 机制实现通道的复用
    //
    // REBLUR_Common_SpatialFilter.hlsli 是一个可重入的代码片段（不是传统意义的头文件）
    // 通过 #define 不同的宏来参数化其行为：
    //
    // REBLUR_SPATIAL_PASS = REBLUR_PRE_PASS: 告诉 SpatialFilter 当前是 Pre-Pass
    //   （还有 REBLUR_BLUR, REBLUR_POST_BLUR 两种模式）
    //
    // 然后分别为 Diffuse 和 Specular 各 #include 一次：
    //   - REBLUR_SPATIAL_LOBE = REBLUR_DIFF: 处理 Diffuse 通道
    //   - REBLUR_SPATIAL_LOBE = REBLUR_SPEC: 处理 Specular 通道
    //
    // 每次 include 后会 #undef 掉 REBLUR_SPATIAL_LOBE 和 MAX_BLUR_RADIUS
    // 这种模式类似于 C++ 模板的编译期参数化
    // =====================================================================================

    // [注解] 定义当前空间滤波模式为 Pre-Pass
    #define REBLUR_SPATIAL_PASS REBLUR_PRE_PASS

    // =====================================================================================
    // [注解] Diffuse 通道的空间预滤波
    //
    // MAX_BLUR_RADIUS = gDiffPrepassBlurRadius: Diffuse 预处理的最大模糊半径
    //   这个值由 CPU 端设置，通常比 Blur/PostBlur pass 的半径小
    //   如果为 0，SpatialFilter 内部会跳过空间滤波（只做 checkerboard resolve）
    // =====================================================================================
    #if( NRD_DIFF )
        #define REBLUR_SPATIAL_LOBE REBLUR_DIFF
        #define MAX_BLUR_RADIUS gDiffPrepassBlurRadius
        #include "REBLUR_Common_SpatialFilter.hlsli"
    #endif

    // =====================================================================================
    // [注解] Specular 通道的空间预滤波
    //
    // MAX_BLUR_RADIUS = gSpecPrepassBlurRadius: Specular 预处理的最大模糊半径
    //
    // Specular 的预处理比 Diffuse 更复杂，因为额外需要：
    //   1. 计算 hit distance for tracking（用于运动估计）
    //   2. 将模糊范围限制在 specular lobe 内（"in-lobe" 模糊）
    //   3. 考虑反射接触点距离的权重衰减
    // =====================================================================================
    #if( NRD_SPEC )
        #define REBLUR_SPATIAL_LOBE REBLUR_SPEC
        #define MAX_BLUR_RADIUS gSpecPrepassBlurRadius
        #include "REBLUR_Common_SpatialFilter.hlsli"
    #endif
}
