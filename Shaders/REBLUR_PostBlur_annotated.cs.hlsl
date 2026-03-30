/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR Post-Blur（后置空间滤波）
//
// 这是 REBLUR 空间阶段里的最后一道 blur pass，位于：
//   TemporalAccumulation / HistoryFix / Blur 之后
//   TemporalStabilization 之前（若启用）
//
// 它的职责可以概括成三点：
//   1. **读取时域累计后的稳定结果**：
//      使用 `gIn_Data1` 中保存的历史长度，推导“当前像素已经有多稳定”
//
//   2. **做最终一轮空间滤波**：
//      继续调用 `REBLUR_Common_SpatialFilter.hlsli`，但以
//      `REBLUR_SPATIAL_PASS = REBLUR_POST_BLUR` 的模式执行，
//      因而会使用 Post-Blur 自己的半径 / 权重参数
//
//   3. **在需要时拷贝 GBuffer / InternalData / 最终副本**：
//      如果没有单独的 TemporalStabilization pass，当前 pass 还会把结果复制到
//      `gOut_*Copy`，供后续直接作为最终稳定输出使用
//
// 这个文件本身非常“薄”：
//   - 它只负责准备中心像素的上下文
//   - 真正的 Poisson 采样、权重计算、Diffuse/Specular 双分支滤波
//     都在 `REBLUR_Common_SpatialFilter.hlsli` 中完成
//
// 可以把它理解为：
//   **PostBlur = 上下文准备器 + 调用两次通用空间滤波器（Diff / Spec）**
// =====================================================================================

// [注解] NRD 核心头文件：导出宏、基础类型、通用 helper
#include "NRD.hlsli"

// [注解] 数学工具库：包含常用近似函数、平滑函数等
#include "ml.hlsli"

// [注解] REBLUR 全局配置：
//        这里会提供 PostBlur 相关的关键宏，例如：
//          - REBLUR_POST_BLUR_ROTATOR_MODE
//          - REBLUR_POST_BLUR_RADIUS_SCALE
//          - REBLUR_POST_BLUR_FRACTION_SCALE
#include "REBLUR_Config.hlsli"

// [注解] 当前 pass 的资源绑定定义：
//        输入：tiles / normal+roughness / data1 / viewZ / diff / spec / SH
//        输出：filtered diff/spec，以及在需要时的 copy / internalData / normalRoughness
#include "REBLUR_PostBlur.resources.hlsli"

// [注解] 通用工具：坐标重建、向量旋转、采样辅助等
#include "Common.hlsli"

// [注解] REBLUR 公共函数：
//        包括 Data1 解包、internalData 打包、hit distance 归一化等
#include "REBLUR_Common.hlsli"

// =====================================================================================
// [注解] 主 Compute Shader 入口
//
// 与 PrePass 一样，PostBlur 也没有使用 shared memory：
//   - 因为真正的采样模式是 Poisson 圆盘 + 旋转核
//   - 访问位置是离散且分散的，不像规则邻域那样适合 preload 到 LDS / shared memory
// =====================================================================================
[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    // [注解] 使用反序 CTA 排布。
    //        宏展开后会生成 `pixelPos` / `threadPos` 等当前线程对应的像素坐标。
    NRD_CTA_ORDER_REVERSED;

    // =====================================================================================
    // [注解] Tile 级 early out
    //
    // `gIn_Tiles` 是 16x16 的 tile 分类结果：
    //   - 非 0 表示该 tile 可以直接视为 sky / 无需降噪
    //   - 这里还顺便过滤掉越界线程
    //
    // 注意返回时 **不写任何输出**，后续必须依赖 viewZ 检查把这些像素排除掉。
    // =====================================================================================
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return; // IMPORTANT: no data output, must be rejected by the "viewZ" check!

    // =====================================================================================
    // [注解] 像素级 early out：超出降噪距离范围的像素不参与 PostBlur
    // =====================================================================================
    float viewZ = UnpackViewZ( gIn_ViewZ[ pixelPos ] );
    if( viewZ > gDenoisingRange )
        return; // IMPORTANT: no data output, must be rejected by the "viewZ" check!

    // =====================================================================================
    // [注解] 中心像素的几何 / 材质属性准备
    //
    // 这些变量不会在本文件里被大量直接使用，
    // 但 `REBLUR_Common_SpatialFilter.hlsli` 会把它们当作“调用者已准备好的上下文”。
    // =====================================================================================
    float materialID;
    float4 normalAndRoughnessPacked = gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ];
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( normalAndRoughnessPacked, materialID );
    float3 N = normalAndRoughness.xyz;                                 // [注解] 世界空间法线
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );      // [注解] 观察空间法线
    float roughness = normalAndRoughness.w;                            // [注解] 中心像素粗糙度

    // [注解] 当前像素中心的 UV 与观察空间位置
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );

    // [注解] 视线方向与 NoV
    //        NoV 会在公共空间滤波器里参与 dominant direction、模糊核形状等计算
    float3 Vv = GetViewVector( Xv, true );
    float NoV = abs( dot( Nv, Vv ) );

    // [注解] 当前深度下的 frustum 尺度：
    //        可理解为“这一层深度里 1 像素在世界空间大约有多大”
    const float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );

    // [注解] 获取 PostBlur 采样核的旋转参数。
    //        PostBlur 默认使用 `REBLUR_POST_BLUR_ROTATOR_MODE`（配置里通常是按帧旋转），
    //        以减少固定采样图案带来的结构化伪影。
    const float4 rotator = GetBlurKernelRotation( REBLUR_POST_BLUR_ROTATOR_MODE, pixelPos, gRotatorPost, gFrameIndex );

    // =====================================================================================
    // [注解] `Data1`：来自 TemporalAccumulation / 前序 pass 的历史长度信息
    //
    // `data1.x` = diffuse accumulation speed
    // `data1.y` = specular accumulation speed
    //
    // PostBlur 不直接使用线性的累积帧数，而是会先把它映射成“非线性累积速度”，
    // 让历史很长的像素在空间滤波时更保守。
    // =====================================================================================
    REBLUR_DATA1_TYPE data1 = UnpackData1( gIn_Data1[ pixelPos ] );

    // =====================================================================================
    // [注解] 输出 GBuffer 副本（可选）
    //
    // 某些管线配置下，后续 pass 还需要读取 normal+roughness，
    // 因此这里会把输入 GBuffer 原样拷贝到输出。
    // =====================================================================================
    #if( REBLUR_COPY_GBUFFER == 1 )
        gOut_Normal_Roughness[ pixelPos ] = normalAndRoughnessPacked;
    #endif

    // =====================================================================================
    // [注解] 当没有 TemporalStabilization pass 时，PostBlur 需要直接输出 InternalData
    //
    // `PackInternalData( diffAccum, specAccum, materialID )` 的结果会在后续历史路径中继续使用。
    // =====================================================================================
    #if( TEMPORAL_STABILIZATION == 0 )
        gOut_InternalData[ pixelPos ] = PackInternalData( data1.x, data1.y, materialID );
    #endif

    // =====================================================================================
    // [注解] 把线性的 accumulation speed 映射成“高级非线性版本”
    //
    // 直觉上：
    //   - 历史越短，说明当前像素还不稳定 → 空间滤波可以积极一些
    //   - 历史越长，说明当前像素已经比较稳定 → 空间滤波应更保守，避免过度抹平细节
    //
    // `GetAdvancedNonLinearAccumSpeed` 就是在做这个映射。
    // =====================================================================================
    float2 nonLinearAccumSpeed;
    nonLinearAccumSpeed.x = GetAdvancedNonLinearAccumSpeed( data1.x );
    nonLinearAccumSpeed.y = GetAdvancedNonLinearAccumSpeed( data1.y );

    #ifdef NRD_COMPILER_DXC
        // =================================================================================
        // [注解] Quad 邻域自适应：
        //        如果相邻 lane 更稳定，就把当前 nonLinearAccumSpeed 往邻域平均值压一压
        //
        // 作用：
        //   - 减少 2x2 quad 内因为历史长度差异过大造成的局部不连续
        //   - 让后续空间滤波核半径在小范围内更平滑
        //
        // `min( current, avg )` 表示：
        //   只允许“变得更保守”，不允许因为邻居更激进而把当前像素拉得更激进
        // =================================================================================
        REBLUR_DATA1_TYPE d10 = QuadReadAcrossX( nonLinearAccumSpeed );
        REBLUR_DATA1_TYPE d01 = QuadReadAcrossY( nonLinearAccumSpeed );

        REBLUR_DATA1_TYPE avg = ( d10 + d01 + nonLinearAccumSpeed ) / 3.0;
        nonLinearAccumSpeed = min( nonLinearAccumSpeed, avg );
    #endif

    // =====================================================================================
    // [注解] 进入公共空间滤波器
    //
    // 下面这段是本文件真正的核心：
    //   1. 把当前 pass 标记为 `REBLUR_POST_BLUR`
    //   2. 然后分别以 Diffuse / Specular 两种 lobe 重新 include 同一个通用滤波文件
    //
    // `REBLUR_Common_SpatialFilter.hlsli` 会直接共享当前函数作用域中的局部变量：
    //   pixelPos, pixelUv, viewZ, N, Nv, Xv, Vv, NoV, roughness,
    //   frustumSize, rotator, nonLinearAccumSpeed, materialID, data1 ...
    //
    // 这也是为什么本文件看起来“没做多少事”，却又必须把这些上下文先全部准备好。
    // =====================================================================================
    #define REBLUR_SPATIAL_PASS REBLUR_POST_BLUR

    #if( NRD_DIFF )
        // [注解] 第一次 include：处理 Diffuse 通道
        //        在公共文件内部会把：
        //          REBLUR_SPATIAL_LOBE = REBLUR_DIFF
        //        映射到 gIn_Diff / gOut_Diff / diffuse 的参数分支。
        #define REBLUR_SPATIAL_LOBE REBLUR_DIFF
        #include "REBLUR_Common_SpatialFilter.hlsli"
    #endif

    #if( NRD_SPEC )
        // [注解] 第二次 include：处理 Specular 通道
        //        内部会切换到 gIn_Spec / gOut_Spec，并启用 roughness / hit distance
        //        相关的 specular 权重逻辑。
        #define REBLUR_SPATIAL_LOBE REBLUR_SPEC
        #include "REBLUR_Common_SpatialFilter.hlsli"
    #endif
}
