/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR Hit Distance Reconstruction（命中距离重建）
//
// 背景知识：
//   REBLUR（Recursive BLock-based Unet Reconstruction）是 NRD 中的一种高质量降噪器。
//   在光线追踪中，每个像素的 "hit distance" 是光线从着色点出发到命中下一个表面的距离。
//   由于采样率低，部分像素的 hit distance 为 0（miss / 无效采样）。
//
//   此 pass 的目的是：对 hit distance 为 0 的像素，利用邻域中有效的 hit distance
//   进行加权插值重建。这是 REBLUR 降噪管线的 **第一个 pass**。
//
// 与 RELAX 版本的核心区别：
//   1. REBLUR 使用 **几何平面距离权重**（而非简单的双边深度权重），更精确地判断
//      两个像素是否属于同一表面
//   2. REBLUR 支持 **hit distance 解压/重压缩**（REBLUR_USE_DECOMPRESSED_HIT_DIST_IN_RECONSTRUCTION）
//      hit distance 在存储时是归一化的，reconstruction 可选择在真实距离空间中操作
//   3. REBLUR 支持 **Occlusion 模式**（NRD_MODE == OCCLUSION），此时只输出标量
//   4. REBLUR 有 **性能模式**（REBLUR_PERFORMANCE_MODE），开启后跳过法线/粗糙度权重
//   5. REBLUR 将 diffuse 和 specular 的权重合并到 float2 中同时处理，更紧凑
// =====================================================================================

// [注解] NRD 核心头文件，公共类型定义、宏、常量
#include "NRD.hlsli"

// [注解] ml.hlsli = Math Library，数学工具函数（AcosApprox, saturate 等）
#include "ml.hlsli"

// [注解] REBLUR 降噪器的配置参数（GROUP_X, GROUP_Y, NRD_BORDER, BUFFER_X, BUFFER_Y 等宏）
//        以及 REBLUR 特有的宏如 REBLUR_USE_DECOMPRESSED_HIT_DIST_IN_RECONSTRUCTION, REBLUR_PERFORMANCE_MODE
#include "REBLUR_Config.hlsli"

// [注解] 自动生成的资源绑定声明，定义了此 shader 使用的所有输入/输出资源
//        包括: gIn_Normal_Roughness, gIn_ViewZ, gIn_Spec, gIn_Diff, gIn_Tiles,
//              gOut_Spec, gOut_Diff 以及各种常量
//        REBLUR 特有常量: gHitDistSettings, gPlaneDistSensitivity, gMinRectDimMulUnproject,
//                         gFrustum, gOrthoMode, gViewToWorld 等
#include "REBLUR_HitDistReconstruction.resources.hlsli"

// [注解] 通用工具函数（坐标变换、打包/解包、权重计算等）
#include "Common.hlsli"

// [注解] REBLUR 系列降噪器的公共工具函数
//        包括 _REBLUR_GetHitDistanceNormalization, ExtractHitDist 等
#include "REBLUR_Common.hlsli"

// =====================================================================================
// [注解] Shared Memory（组共享内存）声明
//
// BUFFER_X / BUFFER_Y = GROUP_X/GROUP_Y + 2 * NRD_BORDER
//   NRD_BORDER 是邻域搜索半径（通常为 1 对应 3x3，或 2 对应 5x5）
//   多出的部分存放边界 halo 区域的数据
// =====================================================================================

// [注解] s_Normal_Roughness: 存储每个像素的 法线(xyz) + 粗糙度(w)
groupshared float4 s_Normal_Roughness[ BUFFER_Y ][ BUFFER_X ];

// [注解] s_HitDist_ViewZ: 存储每个像素的 diffuse hit dist(x) + specular hit dist(y) + viewZ(z)
//        注意：与 RELAX 版本不同，REBLUR 中 x=diffuse, y=specular（RELAX 中 x=spec, y=diff）
groupshared float3 s_HitDist_ViewZ[ BUFFER_Y ][ BUFFER_X ];

// =====================================================================================
// [注解] Preload 函数 - 将全局纹理数据预加载到 shared memory
//
// 参数:
//   sharedPos - 在 shared memory 中的 2D 索引
//   globalPos - 在屏幕/纹理中的 2D 像素坐标
//
// 由 PRELOAD_INTO_SMEM_WITH_TILE_CHECK 宏调用，确保整个 shared memory 被填充
// =====================================================================================
void Preload( uint2 sharedPos, int2 globalPos )
{
    // [注解] 将 globalPos 限制在合法范围 [0, gRectSizeMinusOne] 内
    //        gRectSizeMinusOne = gRectSize - 1，避免越界访问
    //        注意 REBLUR 用 gRectSizeMinusOne 而 RELAX 用 gRectSize - 1.0，效果相同
    globalPos = clamp( globalPos, 0, gRectSizeMinusOne );

    // [注解] 从 gIn_ViewZ 纹理中解包 view-space Z 值（观察空间深度）
    //        viewZ 在此函数中有两个用途：
    //          1. 写入 shared memory 供后续双边权重计算
    //          2. 用于 hit distance 的解压缩（如果启用）
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( globalPos ) ] );

    // [注解] 从 gIn_Normal_Roughness 纹理中解包法线+粗糙度
    //        NRD_FrontEnd_UnpackNormalAndRoughness: 将压缩格式还原为 float4(normal.xyz, roughness)
    //        WithRectOrigin: 加上矩形区域偏移量（支持动态分辨率偏移渲染）
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( globalPos ) ] );

    // [注解] 写入 shared memory（法线+粗糙度）
    s_Normal_Roughness[ sharedPos.y ][ sharedPos.x ] = normalAndRoughness;

    // [注解] hitDist 初始化为 0（与 RELAX 不同，RELAX 初始化为 gDenoisingRange）
    //        float2: x = diffuse hit dist, y = specular hit dist
    float2 hitDist = 0.0;

    // =====================================================================================
    // [注解] Diffuse hit distance 提取
    // =====================================================================================
    #if( NRD_DIFF )
        // [注解] ExtractHitDist: 从输入纹理中提取 hit distance 分量
        //        通常是 gIn_Diff 的 .w 分量（或者 occlusion 模式下的其他分量）
        hitDist.x = ExtractHitDist( gIn_Diff[ globalPos ] );

        // [注解] REBLUR 特有：可选的 hit distance 解压缩
        //        NRD 中 hit distance 默认是归一化存储的（除以一个与 viewZ、材质相关的归一化因子）
        //        如果 REBLUR_USE_DECOMPRESSED_HIT_DIST_IN_RECONSTRUCTION == 1，
        //        则在 preload 时将其乘回归一化因子，恢复为真实的世界空间距离
        //        这样 reconstruction 在真实距离空间中做加权平均，物理上更正确
        //
        //        _REBLUR_GetHitDistanceNormalization(viewZ, settings, roughness):
        //          返回归一化因子 = A + B * pow(max(C, abs(viewZ)), D)
        //          对于 diffuse，roughness 参数传 1.0（diffuse 不依赖粗糙度）
        #if( REBLUR_USE_DECOMPRESSED_HIT_DIST_IN_RECONSTRUCTION == 1 )
            hitDist.x *= _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings.xyz, 1.0 );
        #endif
    #endif

    // =====================================================================================
    // [注解] Specular hit distance 提取
    // =====================================================================================
    #if( NRD_SPEC )
        hitDist.y = ExtractHitDist( gIn_Spec[ globalPos ] );

        // [注解] Specular 的 hit distance 解压缩
        //        与 diffuse 不同，specular 的归一化因子依赖粗糙度（normalAndRoughness.w）
        //        因为 specular 反射的波瓣大小与粗糙度相关：
        //          低粗糙度（镜面）→ hit distance 可以很远
        //          高粗糙度（漫反射状）→ hit distance 通常较短
        #if( REBLUR_USE_DECOMPRESSED_HIT_DIST_IN_RECONSTRUCTION == 1 )
            hitDist.y *= _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings.xyz, normalAndRoughness.w );
        #endif
    #endif

    // [注解] 写入 shared memory: float3(diffHitDist, specHitDist, viewZ)
    s_HitDist_ViewZ[ sharedPos.y ][ sharedPos.x ] = float3( hitDist, viewZ );
}

// =====================================================================================
// [注解] 主 Compute Shader 入口
//
// [numthreads(GROUP_X, GROUP_Y, 1)] 定义线程组尺寸（通常 8x8=64 个线程）
// =====================================================================================
[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    // [注解] NRD_CTA_ORDER_DEFAULT: 从 SV_GroupThreadID / SV_DispatchThreadID 计算出
    //          pixelPos（屏幕像素坐标）和 threadPos（thread group 内的 2D 位置）
    NRD_CTA_ORDER_DEFAULT;

    // =====================================================================================
    // [注解] Preload 阶段
    //
    // gIn_Tiles: tile 级别标记纹理，每 16x16 像素一个 tile
    //   .x 非零表示该 tile 全是天空（不需要降噪）
    //   pixelPos >> 4 = pixelPos / 16，得到 tile 坐标
    //
    // PRELOAD_INTO_SMEM_WITH_TILE_CHECK:
    //   NRD 框架宏，展开后会：
    //   1. 如果是天空 tile 则跳过 preload
    //   2. 否则让每个线程加载 1~多个像素到 shared memory（含 halo 区域）
    //   3. GroupMemoryBarrierWithGroupSync() 同步
    // =====================================================================================
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // [注解] Tile 级别 early out
    //        any(pixelPos > gRectSizeMinusOne): 任一坐标超出有效区域
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // [注解] 从 shared memory 中获取中心像素数据
    //        smemPos = threadPos + NRD_BORDER: 跳过 halo 区域
    int2 smemPos = threadPos + NRD_BORDER;
    float3 center = s_HitDist_ViewZ[ smemPos.y ][ smemPos.x ];
    // [注解] center = float3(diffHitDist, specHitDist, viewZ)

    // [注解] 像素级 early out: viewZ 超出降噪范围（远处/天空）
    if( center.z > gDenoisingRange )
        return;

    // =====================================================================================
    // [注解] 中心像素数据准备
    // =====================================================================================

    // [注解] 从 shared memory 读取法线和粗糙度（这里直接用 smem，不再从全局纹理重读）
    float4 normalAndRoughness = s_Normal_Roughness[ smemPos.y ][ smemPos.x ];
    float3 N = normalAndRoughness.xyz;       // [注解] 世界空间法线
    float roughness = normalAndRoughness.w;   // [注解] 粗糙度

    // [注解] 计算中心像素的 UV 坐标（归一化到 [0,1]，像素中心）
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;

    // [注解] 从 UV + depth 重建观察空间 3D 位置 Xv
    //        Geometry::ReconstructViewPosition: 使用 frustum 参数和 viewZ 反投影
    //        gFrustum: 视锥体参数（near/far/fov 等的紧凑表示）
    //        gOrthoMode: 是否正交投影（0=透视，1=正交）
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, center.z, gOrthoMode );

    // [注解] 将世界空间法线旋转到观察空间
    //        gViewToWorld 的逆变换 = world-to-view 旋转
    //        Nv 是观察空间法线，用于后续几何权重计算
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );

    // [注解] 计算观察方向向量 Vv（从像素位置指向相机原点）
    //        true 参数表示需要归一化
    float3 Vv = GetViewVector( Xv, true );

    // =====================================================================================
    // [注解] 权重参数预计算
    // =====================================================================================

    // [注解] frustumSize: 在当前深度下，一个像素对应的世界空间大小
    //        用于将像素级操作转换为世界空间操作
    //        gMinRectDimMulUnproject: 最小矩形维度 × 反投影因子
    float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, center.z );

    // =====================================================================================
    // [注解] REBLUR 关键区别 #1: 几何平面距离权重参数
    //
    // 与 RELAX 使用简单的 GetBilateralWeight(sampleViewZ, centerViewZ) 不同，
    // REBLUR 使用基于 **平面距离** 的权重：
    //   对于邻域像素位置 Xs，计算它到中心像素所在平面的距离 = dot(Nv, Xs)
    //   如果距离大（说明邻域像素不在同一平面/表面上），权重降低
    //
    // 这比单纯的深度差异权重更精确：
    //   - 深度权重：只看 Z 差异，在倾斜表面上可能误判
    //   - 平面距离权重：沿法线方向测量，能正确处理倾斜表面
    //
    // gPlaneDistSensitivity: 平面距离敏感度参数，控制权重衰减速度
    // =====================================================================================
    float2 geometryWeightParams = GetGeometryWeightParams( gPlaneDistSensitivity, frustumSize, Xv, Nv );

    // [注解] 粗糙度权重参数（与 RELAX 相同）
    //        roughness² 作为输入（感知线性空间）
    float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams( roughness * roughness );

    // [注解] Diffuse 法线权重参数：不依赖粗糙度
    float diffNormalWeightParam = GetNormalWeightParam( 1.0, 1.0 );

    // [注解] Specular 法线权重参数：依赖粗糙度（低粗糙度 → 更严格的法线一致性要求）
    float specNormalWeightParam = GetNormalWeightParam( 1.0, 1.0, roughness );

    // =====================================================================================
    // [注解] 加权求和初始化
    //
    // sum = float2(diffWeight, specWeight)
    //   如果中心的 hit distance != 0（有效），给权重 1000（绝对主导）
    //   如果中心的 hit distance == 0（无效），权重为 0（完全依赖邻域重建）
    //
    // center.xy 同时存储两个通道的 hit distance 累加值
    //   乘以 sum 后：有效值 → center * 1000；无效值 → 0
    //
    // 注意：这里用 float2 同时处理 diff 和 spec 两个通道，比 RELAX 的分开处理更紧凑
    // =====================================================================================
    float2 sum = 1000.0 * float2( center.xy != 0.0 );
    center.xy *= sum;

    // =====================================================================================
    // [注解] 邻域遍历与加权求和
    //
    // 遍历以中心像素为中心、半径为 NRD_BORDER 的正方形邻域
    // NRD_BORDER = 1 → 3x3；NRD_BORDER = 2 → 5x5（REBLUR 可配置更大的搜索范围）
    // =====================================================================================
    [unroll]  // [注解] 编译器展开循环，提高 GPU 性能
    for( j = 0; j <= NRD_BORDER * 2; j++ )
    {
        [unroll]
        for( i = 0; i <= NRD_BORDER * 2; i++ )
        {
            // [注解] o = 相对于中心像素的偏移量（浮点，用于后续 UV 计算）
            float2 o = float2( i, j ) - NRD_BORDER;

            // [注解] 跳过中心像素自身
            if( o.x == 0.0 && o.y == 0.0 )
                continue;

            // [注解] 从 shared memory 读取邻域像素数据
            //        data = float3(diffHitDist, specHitDist, viewZ)
            int2 pos = threadPos + int2( i, j );
            float3 data = s_HitDist_ViewZ[ pos.y ][ pos.x ];

            // =====================================================================================
            // [注解] 基础权重计算
            // =====================================================================================

            // [注解] 1. 屏幕边界检查：邻域 UV 是否在 [0,1] 范围内
            float w = IsInScreenNearest( pixelUv + o * gRectSizeInv );

            // [注解] 2. 高斯空间权重：基于像素距离的高斯衰减
            w *= GetGaussianWeight( length( o ) * 0.5 );

            // =====================================================================================
            // [注解] REBLUR 关键区别 #2: 几何平面距离权重（替代简单的双边深度权重）
            //
            // 原理：
            //   1. 从邻域像素的 UV + viewZ 重建其观察空间 3D 位置 Xvs
            //   2. 计算 NoX = dot(Nv, Xvs) = 邻域像素在中心法线方向上的投影
            //   3. ComputeWeight(NoX, params) 是 **严格权重**（非指数衰减）
            //      如果 NoX 与中心像素的 NoX 差异大（不在同一平面），权重直接截断为 0
            //
            // 注释原文："This weight is strict (non exponential) because we need to avoid
            //            accessing data from other surfaces"
            //   严格截断权重比指数衰减更安全 —— 确保绝不会混入不同表面的数据
            // =====================================================================================
            float2 uv = pixelUv + o * gRectSizeInv;
            float3 Xvs = Geometry::ReconstructViewPosition( uv, gFrustum, data.z, gOrthoMode );
            float NoX = dot( Nv, Xvs );

            // [注解] 3. 几何平面距离权重（严格截断）
            w *= ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );

            // [注解] 4. 深度有效性检查：如果邻域 viewZ 超出降噪范围，权重设为 0
            //        注释说明：当 data.z 超出范围时 |NoX| 可能接近 0，
            //        这会导致几何权重计算不可靠，所以显式设为 0
            w = data.z < gDenoisingRange ? w : 0.0;

            // =====================================================================================
            // [注解] REBLUR 关键区别 #3: 性能模式开关
            //
            // ww = float2(diffWeight, specWeight)，两个通道同时处理
            //
            // REBLUR_PERFORMANCE_MODE == 0（质量模式）:
            //   额外计算法线权重和粗糙度权重，结果更精确
            //
            // REBLUR_PERFORMANCE_MODE == 1（性能模式）:
            //   跳过这些额外权重，只用基础的几何+高斯权重
            //   适用于对性能要求高的场景
            //
            // RELAX 版本没有这个性能模式开关
            // =====================================================================================
            float2 ww = w;
            #if( REBLUR_PERFORMANCE_MODE == 0 )
                // [注解] 质量模式下：读取邻域法线+粗糙度
                float4 normalAndRoughness = s_Normal_Roughness[ pos.y ][ pos.x ];

                // [注解] 计算中心法线与邻域法线的夹角
                float cosa = dot( N, normalAndRoughness.xyz );
                float angle = Math::AcosApprox( cosa );

                // [注解] 注释原文：
                //   "These weights have infinite exponential tails, because with strict
                //    weights we are reducing a chance to find a valid sample in 3x3 or 5x5 area"
                //
                //   法线/粗糙度权重使用指数衰减（有无限长尾），而非严格截断
                //   因为几何权重已经做了严格截断，如果法线权重也截断，
                //   在 3x3 或 5x5 区域内可能找不到任何有效样本
                //   指数衰减保证即使法线差异较大，仍有微小概率被选中

                // [注解] 5. Diffuse 法线权重（指数衰减）
                ww.x *= ComputeExponentialWeight( angle, diffNormalWeightParam, 0.0 );

                // [注解] 6. Specular 法线权重（指数衰减，受粗糙度影响）
                ww.y *= ComputeExponentialWeight( angle, specNormalWeightParam, 0.0 );

                // [注解] 7. Specular 粗糙度权重：邻域粗糙度²与中心粗糙度²的匹配度
                ww.y *= ComputeExponentialWeight( normalAndRoughness.w * normalAndRoughness.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y );
            #endif

            // [注解] NaN 防护：如果权重为 0 或 NaN，将 hit distance 设为 0
            data.x = Denanify( ww.x, data.x );
            data.y = Denanify( ww.y, data.y );

            // [注解] 有效性权重：hit distance 为 0 的邻域像素不参与累加
            ww *= float2( data.xy != 0.0 );

            // [注解] 累加：加权 hit distance 和权重（两个通道同时处理）
            center.xy += data.xy * ww;
            sum += ww;
        }
    }

    // =====================================================================================
    // [注解] 归一化加权和
    //
    // center.xy / sum = 加权平均的 hit distance
    // max(sum, NRD_EPS): 防止除以 0
    //   注释说明："if all conditions are met, sum can't be 0"
    //   即正常情况下，至少中心像素自身或某个邻域像素会贡献权重
    //   NRD_EPS 只是安全保障
    // =====================================================================================
    center.xy /= max( sum, NRD_EPS );

    // =====================================================================================
    // [注解] REBLUR 关键区别 #4: Hit distance 重新归一化
    //
    // 如果之前在 Preload 中做了解压缩（乘以归一化因子），
    // 这里需要除回去，将重建后的 hit distance 重新归一化存储
    //
    // 为什么要这样做？
    //   在真实距离空间中做加权平均更物理正确（因为不同深度的归一化因子不同，
    //   在归一化空间中直接平均会引入误差）
    //   但最终存储仍需要归一化格式，因为后续 pass 期望输入是归一化的
    //
    // center.x (diffuse): 归一化因子与 roughness=1.0 对应
    // center.y (specular): 归一化因子与当前 roughness 对应
    // center.z = viewZ: 用于计算归一化因子
    // =====================================================================================
    #if( REBLUR_USE_DECOMPRESSED_HIT_DIST_IN_RECONSTRUCTION == 1 )
        center.x /= _REBLUR_GetHitDistanceNormalization( center.z, gHitDistSettings.xyz, 1.0 );
        center.y /= _REBLUR_GetHitDistanceNormalization( center.z, gHitDistSettings.xyz, roughness );
    #endif

    // =====================================================================================
    // [注解] 输出阶段
    //
    // REBLUR 关键区别 #5: 支持 Occlusion 模式
    //   - NRD_MODE == OCCLUSION: 只输出标量 hit distance（用于 ambient occlusion）
    //   - 其他模式: 输出 float4(radiance.rgb, hitDist)
    // =====================================================================================
    #if( NRD_DIFF )
        #if( NRD_MODE == OCCLUSION )
            // [注解] Occlusion 模式：只输出 diffuse hit distance 标量
            //        用于 diffuse occlusion 降噪
            gOut_Diff[ pixelPos ] = center.x;
        #else
            // [注解] 正常模式：RGB = 原始 diffuse 辐照度（直接从输入读取），A = 重建后的 hit distance
            float3 diff = gIn_Diff[ pixelPos ].xyz;
            gOut_Diff[ pixelPos ] = float4( diff, center.x );
        #endif
    #endif

    #if( NRD_SPEC )
        #if( NRD_MODE == OCCLUSION )
            // [注解] Occlusion 模式：只输出 specular hit distance 标量
            //        用于 specular occlusion 降噪
            gOut_Spec[ pixelPos ] = center.y;
        #else
            // [注解] 正常模式：RGB = 原始 specular 辐照度，A = 重建后的 hit distance
            float3 spec = gIn_Spec[ pixelPos ].xyz;
            gOut_Spec[ pixelPos ] = float4( spec, center.y );
        #endif
    #endif
}
