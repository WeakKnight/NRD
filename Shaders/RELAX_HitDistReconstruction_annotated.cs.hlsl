/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：Hit Distance Reconstruction（命中距离重建）
//
// 背景知识：
//   在光线追踪降噪中，每个像素会得到一个 "hit distance"（光线从表面出发到命中下一个
//   表面的距离）。但由于光追采样率低，某些像素可能没有有效的 hit distance（值为 0，
//   表示 miss 或无效采样）。
//
//   此 pass 的目的是：对那些没有有效 hit distance 的像素，利用邻域中有效的 hit distance
//   进行加权插值重建，从而为后续降噪 pass 提供更完整的 hit distance 信息。
//
//   这是 RELAX 降噪管线的第一个 pass，在正式的时空降噪之前执行。
// =====================================================================================

// [注解] 引入 NRD 核心头文件，包含公共类型定义、宏、常量等
#include "NRD.hlsli"

// [注解] ml.hlsli = Math Library，包含数学工具函数（如 AcosApprox, saturate 等）
#include "ml.hlsli"

// [注解] RELAX 降噪器的配置参数（如 GROUP_X, GROUP_Y, NRD_BORDER, BUFFER_X, BUFFER_Y 等宏定义）
#include "RELAX_Config.hlsli"

// [注解] 自动生成的资源绑定声明，定义了此 shader 使用的所有输入/输出资源
//        包括: gIn_Normal_Roughness, gIn_ViewZ, gIn_Spec, gIn_Diff, gIn_Tiles,
//              gOut_Spec, gOut_Diff 以及各种常量 (gRectSize, gDenoisingRange 等)
#include "RELAX_HitDistReconstruction.resources.hlsli"

// [注解] 通用工具函数（坐标变换、打包/解包、权重计算等）
#include "Common.hlsli"

// [注解] RELAX 系列降噪器的公共工具函数
#include "RELAX_Common.hlsli"

// =====================================================================================
// [注解] Shared Memory（共享内存 / Group Shared Memory）声明
//
// 为什么用 shared memory？
//   在 compute shader 中，同一个 thread group 的线程可以共享数据。
//   Hit distance reconstruction 需要读取邻域像素的法线、粗糙度、hit distance、viewZ，
//   如果每个线程都直接从全局纹理中多次采样，性能很差。
//   所以先将数据 preload 到 shared memory，后续邻域访问就只需读 shared memory，
//   大大减少全局内存访问次数。
//
// BUFFER_X / BUFFER_Y = GROUP_X/GROUP_Y + 2 * NRD_BORDER
//   其中 NRD_BORDER 是邻域搜索半径（通常为 1），多出的部分存放边界 halo 区域的数据。
// =====================================================================================

// [注解] s_Normal_Roughness: 存储每个像素的 法线(xyz) + 粗糙度(w)
groupshared float4 s_Normal_Roughness[BUFFER_Y][BUFFER_X];

// [注解] s_HitDist_ViewZ: 存储每个像素的 specular hit distance(x) + diffuse hit distance(y) + viewZ(z)
groupshared float3 s_HitDist_ViewZ[BUFFER_Y][BUFFER_X];

// =====================================================================================
// [注解] Preload 函数 - 将全局纹理数据预加载到 shared memory
//
// 参数:
//   sharedPos - 在 shared memory 中的 2D 索引
//   globalPos - 在屏幕/纹理中的 2D 像素坐标
//
// 这个函数会被 PRELOAD_INTO_SMEM_WITH_TILE_CHECK 宏展开后多次调用，
// 确保整个 shared memory（包括 halo 区域）都被填充。
// =====================================================================================
void Preload(uint2 sharedPos, int2 globalPos)
{
    // [注解] 将 globalPos 限制在合法范围 [0, gRectSize-1] 内
    //        gRectSize 是降噪输入的矩形区域大小（可以小于全屏分辨率，适配动态分辨率）
    //        clamp 保证越界时读取边缘像素（mirror/clamp 边界处理）
    globalPos = clamp(globalPos, 0, gRectSize - 1.0);

    // [注解] 从 gIn_Normal_Roughness 纹理中读取并解包法线+粗糙度
    //        NRD_FrontEnd_UnpackNormalAndRoughness: 将压缩格式还原为 float4(normal.xyz, roughness)
    //        WithRectOrigin: 将矩形区域坐标加上偏移量（支持动态分辨率中的偏移渲染）
    //        注意：这里故意不使用 materialID，因为 hit distance reconstruction 不需要材质区分
    // It's ok that we don't use materialID in Hitdist reconstruction
    float4 normalRoughness = NRD_FrontEnd_UnpackNormalAndRoughness(gIn_Normal_Roughness[WithRectOrigin(globalPos)]);

    // [注解] 从 gIn_ViewZ 纹理中读取并解包 view-space Z 值（观察空间深度）
    //        viewZ 用于双边权重计算（深度差异大的像素权重低）
    float viewZ = UnpackViewZ(gIn_ViewZ[WithRectOrigin(globalPos)]);

    // [注解] hitDist 初始化为 gDenoisingRange（降噪有效范围的最大值）
    //        这是一个 float2: x = specular hit dist, y = diffuse hit dist
    //        如果相应通道没有启用，就保持这个默认最大值
    float2 hitDist = gDenoisingRange;

    // [注解] 条件编译：如果启用了 Specular 通道（NRD_SPEC=1）
    //        从 gIn_Spec 纹理的 .w 分量读取 specular hit distance
    //        注意 gIn_Spec 的格式是 float4(specularRadiance.rgb, hitDist)
    #if( NRD_SPEC )
        hitDist.x = gIn_Spec[globalPos].w;
    #endif

    // [注解] 条件编译：如果启用了 Diffuse 通道（NRD_DIFF=1）
    //        从 gIn_Diff 纹理的 .w 分量读取 diffuse hit distance
    //        注意 gIn_Diff 的格式是 float4(diffuseRadiance.rgb, hitDist)
    #if( NRD_DIFF )
        hitDist.y = gIn_Diff[globalPos].w;
    #endif

    // [注解] 写入 shared memory
    //        s_Normal_Roughness: 存 法线(xyz) + 粗糙度(w)
    //        s_HitDist_ViewZ: 存 specHitDist(x) + diffHitDist(y) + viewZ(z)
    s_Normal_Roughness[sharedPos.y][sharedPos.x] = normalRoughness;
    s_HitDist_ViewZ[sharedPos.y][sharedPos.x] = float3(hitDist, viewZ);
}

// =====================================================================================
// [注解] 主 Compute Shader 入口
//
// [numthreads(GROUP_X, GROUP_Y, 1)] 定义了线程组的尺寸
//   - 通常 GROUP_X = GROUP_Y = 8，即每个 thread group 有 8x8=64 个线程
//   - 每个线程处理一个像素
//
// NRD_EXPORT: 导出宏，确保函数可被 NRD 框架调用
// NRD_CS_MAIN: compute shader 主函数名宏
// NRD_CS_MAIN_ARGS: 展开为标准 CS 参数（SV_GroupThreadID, SV_DispatchThreadID 等）
// =====================================================================================
[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    // [注解] NRD_CTA_ORDER_DEFAULT: 定义线程组内的线程排列顺序宏
    //        展开后会从 SV_GroupThreadID / SV_DispatchThreadID 计算出:
    //          - pixelPos: 当前线程对应的屏幕像素坐标
    //          - threadPos: 当前线程在 thread group 内的 2D 位置
    NRD_CTA_ORDER_DEFAULT;

    // [注解] 计算当前像素的 UV 坐标（归一化到 [0,1]）
    //        +0.5 是为了取像素中心（像素坐标是整数，加 0.5 变为中心点）
    //        gRectSizeInv = 1.0 / gRectSize
    float2 pixelUv = float2(pixelPos + 0.5) * gRectSizeInv;

    // =====================================================================================
    // [注解] Preload 阶段 - 将数据预加载到 shared memory
    //
    // gIn_Tiles: 一个 tile 级别的标记纹理，每 16x16 像素一个 tile
    //   值非零表示该 tile 全是天空像素（不需要降噪）
    //   pixelPos >> 4 等价于 pixelPos / 16，得到 tile 坐标
    //
    // PRELOAD_INTO_SMEM_WITH_TILE_CHECK:
    //   这是一个 NRD 框架宏，展开后会：
    //   1. 检查 isSky，如果是天空 tile 则跳过 preload（性能优化）
    //   2. 否则，当前线程负责加载 1~多个像素数据到 shared memory
    //      （包括 halo 区域，确保 BUFFER_X × BUFFER_Y 全部覆盖）
    //   3. 调用 GroupMemoryBarrierWithGroupSync() 保证同步
    // =====================================================================================
    float isSky = gIn_Tiles[pixelPos >> 4];
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // [注解] Tile 级别的 early out（提前退出）
    //        如果当前 tile 全是天空（isSky != 0），或者当前像素超出有效区域，则直接返回
    //        天空像素没有几何表面，不需要 hit distance reconstruction
    if (isSky != 0.0 || pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y)
        return;

    // [注解] 从 shared memory 中获取当前像素（中心像素）的数据
    //        smemPos = threadPos + NRD_BORDER: 加上 border 偏移是因为 shared memory
    //        的 [0,0] 位置存的是 halo 区域的数据，实际线程 (0,0) 对应 smem 中的 (NRD_BORDER, NRD_BORDER)
    int2 smemPos = threadPos + NRD_BORDER;
    float3 centerHitdistViewZ = s_HitDist_ViewZ[smemPos.y][smemPos.x];

    // [注解] 取出中心像素的 view-space Z（观察空间深度）
    float centerViewZ = centerHitdistViewZ.z;

    // [注解] 像素级别的 early out
    //        如果 viewZ 超过了降噪范围（gDenoisingRange），说明该像素在远处/天空，无需处理
    if (centerViewZ > gDenoisingRange)
        return;

    // =====================================================================================
    // [注解] 读取中心像素的法线和粗糙度（从全局纹理再读一次，因为需要完整精度）
    //        shared memory 中已经有了，但这里直接读原始纹理可能是为了精度或一致性
    // =====================================================================================
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness(gIn_Normal_Roughness[WithRectOrigin( pixelPos )]);

    // [注解] 提取法线 (xyz) 和粗糙度 (w)
    float3 centerNormal = normalAndRoughness.xyz;
    float centerRoughness = normalAndRoughness.w;

    // =====================================================================================
    // [注解] ==== Specular Hit Distance Reconstruction ====
    //        只在 NRD_SPEC 启用时编译
    // =====================================================================================
#if( NRD_SPEC )
    // [注解] 中心像素的 specular 辐照度 (RGB)，后续只重建 .w (hit distance)，辐照度直接传递
    float3 centerSpecularIllumination = gIn_Spec[pixelPos].xyz;

    // [注解] 中心像素的 specular hit distance
    float centerSpecularHitDist = centerHitdistViewZ.x;

    // [注解] 获取基于粗糙度的 "宽松粗糙度权重" 参数
    //        centerRoughness * centerRoughness: 使用粗糙度的平方（通常更接近感知线性）
    //        这个权重用于判断邻域像素的粗糙度是否与中心像素相似
    //        粗糙度差异大 → 权重低（不同材质特性的像素不应互相影响）
    float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams(centerRoughness * centerRoughness);

    // [注解] 获取基于法线的权重参数（用于 specular 通道）
    //        参数 (1.0, 1.0, centerRoughness): 对于 specular，法线权重会考虑粗糙度
    //        粗糙度高时，法线差异的容忍度更高（因为 glossy/diffuse-like 反射更分散）
    //        粗糙度低时（镜面反射），法线必须非常一致
    float specularNormalWeightParam = GetNormalWeightParam(1.0, 1.0, centerRoughness);

    // [注解] 初始化加权求和的累加器
    //        如果中心像素的 hit distance != 0（有效采样），给它一个很大的权重 1000.0
    //        这样中心像素的值在加权平均中占绝对主导地位（只在自身无效时才依赖邻域重建）
    //        如果中心 hit distance == 0（无效/miss），权重为 0，完全依赖邻域
    float sumSpecularWeight = 1000.0 * float(centerSpecularHitDist != 0.0);
    float sumSpecularHitDist = centerSpecularHitDist * sumSpecularWeight;
#endif

    // =====================================================================================
    // [注解] ==== Diffuse Hit Distance Reconstruction ====
    //        只在 NRD_DIFF 启用时编译，逻辑与 specular 类似
    // =====================================================================================
#if( NRD_DIFF )
    // [注解] 中心像素的 diffuse 辐照度 (RGB)
    float3 centerDiffuseIllumination = gIn_Diff[pixelPos].xyz;

    // [注解] 中心像素的 diffuse hit distance
    float centerDiffuseHitDist = centerHitdistViewZ.y;

    // [注解] Diffuse 的法线权重参数，不需要粗糙度（diffuse 反射与粗糙度无关）
    float diffuseNormalWeightParam = GetNormalWeightParam(1.0, 1.0);

    // [注解] 同样的初始化策略：中心有效 → 权重 1000；中心无效 → 权重 0
    float sumDiffuseWeight = 1000.0 * float(centerDiffuseHitDist != 0.0);
    float sumDiffuseHitDist = centerDiffuseHitDist * sumDiffuseWeight;
#endif

    // =====================================================================================
    // [注解] ==== 邻域遍历与加权求和 ====
    //
    // 遍历以中心像素为中心、半径为 NRD_BORDER 的正方形邻域
    // NRD_BORDER 通常为 1，所以遍历范围是 3x3 = 9 个像素（跳过中心自身）
    //
    // 对每个邻域像素，计算多个权重因子并组合：
    //   1. 屏幕边界检查权重
    //   2. 深度有效性权重（viewZ < gDenoisingRange）
    //   3. 高斯空间距离权重（距离越远权重越小）
    //   4. 双边深度权重（深度差异大 → 权重低）
    //   5. 法线权重（法线差异大 → 权重低）
    //   6. 粗糙度权重（仅 specular）
    //   7. 有效性权重（hit distance 为 0 的邻域像素权重为 0）
    // =====================================================================================
    [unroll]  // [注解] 编译器展开循环，提高 GPU 性能（循环次数编译时已知）
    for (int dy = 0; dy <= NRD_BORDER * 2; dy++)
    {
        [unroll]
        for (int dx = 0; dx <= NRD_BORDER * 2; dx++)
        {
            // [注解] o 是相对于中心像素的偏移量
            //        当 NRD_BORDER=1 时, o ∈ {(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)}
            int2 o = int2(dx, dy) - NRD_BORDER;

            // [注解] 跳过中心像素自身（已经在累加器初始化时处理了）
            if (o.x == 0 && o.y == 0)
                continue;

            // [注解] 计算在 shared memory 中的位置
            //        pos = threadPos + (dx, dy)，因为 shared memory 的布局已经包含了 border
            int2 pos = threadPos + int2(dx, dy);

            // [注解] 从 shared memory 读取邻域像素的法线 + 粗糙度
            float4 sampleNormalRoughness = s_Normal_Roughness[pos.y][pos.x];
            float3 sampleNormal = sampleNormalRoughness.xyz;

            // [注解] 注意：这里类型写成 float3 是源码的一个小 quirk，实际只用标量值
            //        sampleNormalRoughness.w 是粗糙度标量
            float3 sampleRoughness = sampleNormalRoughness.w;

            // [注解] 从 shared memory 读取邻域像素的 hit distance + viewZ
            float3 sampleHitdistViewZ = s_HitDist_ViewZ[pos.y][pos.x];
            float sampleViewZ = sampleHitdistViewZ.z;

            // [注解] 计算中心法线与邻域法线的夹角
            //        dot → cos(angle)，然后通过 AcosApprox 近似求反余弦得到角度
            float cosa = dot(centerNormal, sampleNormal);
            float angle = Math::AcosApprox(cosa);

            // =====================================================================================
            // [注解] 计算基础权重 w（specular 和 diffuse 共用）
            // =====================================================================================

            // [注解] 1. 屏幕边界检查：邻域像素的 UV 是否还在屏幕内
            //        pixelUv + o * gRectSizeInv = 邻域像素的 UV 坐标
            //        如果超出 [0,1] 范围，返回 0（不参与加权）
            float w = IsInScreenNearest(pixelUv + o * gRectSizeInv);

            // [注解] 2. 深度有效性：邻域像素的 viewZ 必须在降噪范围内
            w *= float(sampleViewZ < gDenoisingRange);

            // [注解] 3. 高斯空间权重：基于空间距离的高斯衰减
            //        length(o) 是偏移量的欧几里得距离
            //        乘以 0.5 是缩放因子（控制高斯核的宽度）
            w *= GetGaussianWeight(length(o) * 0.5);

            // [注解] 4. 双边深度权重：邻域像素与中心像素的深度差异越大，权重越小
            //        这是双边滤波的核心思想——保边滤波
            //        防止跨越深度不连续处（物体边缘）的错误混合
            w *= GetBilateralWeight(sampleViewZ, centerViewZ);

            // =====================================================================================
            // [注解] ==== Specular 通道的额外权重 ====
            // =====================================================================================
#if( NRD_SPEC )
            float specularWeight = w;

            // [注解] 5. 法线权重（specular 版本）：法线差异大 → 权重低
            //        specularNormalWeightParam 控制衰减速度，受粗糙度影响
            //        ComputeExponentialWeight(value, param, center):
            //          计算 exp(-param * |value - center|) 形式的权重
            specularWeight *= ComputeExponentialWeight(angle, specularNormalWeightParam, 0.0);

            // [注解] 6. 粗糙度权重：邻域像素的粗糙度是否与中心像素匹配
            //        normalAndRoughness.w * normalAndRoughness.w = centerRoughness²
            //        relaxedRoughnessWeightParams: (x=param, y=center) 来自之前的计算
            //        注意：这里用的是中心像素的粗糙度² 而非邻域的粗糙度
            //        这看起来像是一个简化：只要中心粗糙度足够高，就放松权重限制
            specularWeight *= ComputeExponentialWeight(normalAndRoughness.w * normalAndRoughness.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y);

            // [注解] 从 shared memory 读取邻域像素的 specular hit distance
            float sampleSpecularHitDist = sampleHitdistViewZ.x;

            // [注解] Denanify: 如果 specularWeight 为 0 或 NaN，则将 sampleSpecularHitDist 设为 0
            //        这是一个安全措施，防止 NaN 污染后续计算
            sampleSpecularHitDist = Denanify( specularWeight, sampleSpecularHitDist );

            // [注解] 7. 有效性权重：如果邻域的 hit distance 为 0（无效采样），权重设为 0
            //        只有有效的 hit distance 才参与重建
            specularWeight *= float(sampleSpecularHitDist != 0.0);

            // [注解] 累加加权 hit distance 和权重
            sumSpecularHitDist += sampleSpecularHitDist * specularWeight;
            sumSpecularWeight += specularWeight;
#endif

            // =====================================================================================
            // [注解] ==== Diffuse 通道的额外权重 ====
            //        逻辑与 specular 类似，但更简单（不需要粗糙度权重）
            // =====================================================================================
#if( NRD_DIFF )
            float diffuseWeight = w;

            // [注解] 5. 法线权重（diffuse 版本）：不受粗糙度影响
            //        diffuse 反射只依赖法线方向，与粗糙度无关
            diffuseWeight *= ComputeExponentialWeight(angle, diffuseNormalWeightParam, 0.0);

            // [注解] 从 shared memory 读取邻域像素的 diffuse hit distance
            float sampleDiffuseHitDist = sampleHitdistViewZ.y;

            // [注解] 同样的 NaN 防护
            sampleDiffuseHitDist = Denanify( diffuseWeight, sampleDiffuseHitDist );

            // [注解] 有效性权重
            diffuseWeight *= float(sampleDiffuseHitDist != 0.0);

            // [注解] 累加加权 hit distance 和权重
            //        注意：diffuse 这里有个额外的安全检查 diffuseWeight == 0.0 ? 0.0 : ...
            //        这是为了在权重为 0 时确保不累加任何值（避免浮点精度问题）
            sumDiffuseHitDist += diffuseWeight == 0.0 ? 0.0 : sampleDiffuseHitDist * diffuseWeight;
            sumDiffuseWeight += diffuseWeight;
#endif
        }
    }

    // =====================================================================================
    // [注解] ==== 输出阶段 ====
    //
    // 将加权平均后的 hit distance 写回输出纹理
    // RGB 保持原始辐照度不变，只替换 .w 分量为重建后的 hit distance
    //
    // 除以 max(weight, 1e-6) 是为了：
    //   1. 计算加权平均（总和 / 总权重）
    //   2. 防止除以 0（用 1e-6 作为最小值）
    //
    // 特殊情况：
    //   - 中心有效 + 邻域有效 → 以中心为主的加权平均（中心权重 1000 远大于邻域权重）
    //   - 中心无效 + 邻域有效 → 纯邻域加权平均（重建）
    //   - 中心有效 + 邻域无效 → 直接使用中心值（只有中心贡献了权重）
    //   - 中心无效 + 邻域无效 → 0/1e-6 ≈ 0（hit distance 为 0，后续 pass 会处理）
    // =====================================================================================

#if( NRD_SPEC )
    // [注解] Specular 输出：RGB = 原始 specular 辐照度，A = 重建后的 hit distance
    sumSpecularHitDist /= max(sumSpecularWeight, 1e-6);
    gOut_Spec[pixelPos] = float4(centerSpecularIllumination, sumSpecularHitDist);
#endif

#if( NRD_DIFF )
    // [注解] Diffuse 输出：RGB = 原始 diffuse 辐照度，A = 重建后的 hit distance
    sumDiffuseHitDist /= max(sumDiffuseWeight, 1e-6);
    gOut_Diff[pixelPos] = float4(centerDiffuseIllumination, sumDiffuseHitDist);
#endif

}
