/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此文件的核心功能：REBLUR 通用空间滤波器（Spatial Filter）
//
// 这是一个 **可重入的代码片段**（不是传统头文件），通过宏参数化后被多次 #include：
//   - 被 REBLUR_PrePass.cs.hlsl  include 两次（Diff + Spec）
//   - 被 REBLUR_Blur.cs.hlsl     include 两次（Diff + Spec）
//   - 被 REBLUR_PostBlur.cs.hlsl include 两次（Diff + Spec）
//
// 通过以下宏来参数化行为：
//   REBLUR_SPATIAL_PASS: 当前是哪个 pass（REBLUR_PRE_PASS / REBLUR_BLUR / REBLUR_POST_BLUR）
//   REBLUR_SPATIAL_LOBE: 当前处理哪个通道（REBLUR_DIFF / REBLUR_SPEC）
//   MAX_BLUR_RADIUS:     最大模糊半径
//
// 核心算法：
//   使用 Poisson 采样盘在空间域做加权模糊，权重综合考虑：
//   几何平面距离、法线一致性、粗糙度一致性、材质 ID、hit distance 一致性
//
// 此文件结尾会 #undef 所有参数宏，允许被再次 include（不同参数）
// =====================================================================================

// =====================================================================================
// [注解] 编译期参数校验
//        REBLUR_SPATIAL_PASS 和 REBLUR_SPATIAL_LOBE 必须在 include 前 #define
// =====================================================================================
#ifndef REBLUR_SPATIAL_PASS
    #error REBLUR_SPATIAL_PASS must be defined!
#endif

#ifndef REBLUR_SPATIAL_LOBE
    #error REBLUR_SPATIAL_LOBE must be defined!
#endif

// [注解] 如果调用方没有定义 MAX_BLUR_RADIUS，使用默认的 gMaxBlurRadius
#ifndef MAX_BLUR_RADIUS
    #define MAX_BLUR_RADIUS             gMaxBlurRadius
#endif

// =====================================================================================
// [注解] 根据 pass 类型选择 Poisson 采样点数量和采样模式
//
// Pre-Pass: 使用较少的采样点（轻量级模糊，主要目的是 checkerboard resolve + 初始降噪）
// Blur/PostBlur: 使用更多采样点（更强的空间降噪）
// =====================================================================================
#if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
    #define POISSON_SAMPLE_NUM          REBLUR_PRE_PASS_POISSON_SAMPLE_NUM
    #define POISSON_SAMPLES( i )        REBLUR_PRE_PASS_POISSON_SAMPLES( i )
#else
    #define POISSON_SAMPLE_NUM          REBLUR_POISSON_SAMPLE_NUM
    #define POISSON_SAMPLES( i )        REBLUR_POISSON_SAMPLES( i )
#endif

// =====================================================================================
// [注解] 根据通道类型（Diffuse / Specular）映射输入/输出资源和参数
//
// 这是此文件能被复用两次的关键 —— 通过宏将抽象名称映射到具体资源
//
// Diffuse 通道 (REBLUR_DIFF):
//   - USE_SCREEN_SPACE: 是否使用屏幕空间采样（vs 世界空间采样）
//   - ROUGHNESS = 1.0: Diffuse 等价于粗糙度为 1 的材质
//   - INPUT/OUTPUT: gIn_Diff / gOut_Diff 等
//
// Specular 通道 (REBLUR_SPEC):
//   - ROUGHNESS = roughness: 使用实际粗糙度
//   - INPUT/OUTPUT: gIn_Spec / gOut_Spec 等
// =====================================================================================
#if( REBLUR_SPATIAL_LOBE == REBLUR_DIFF )
    #define USE_SCREEN_SPACE            REBLUR_USE_SCREEN_SPACE_SAMPLING_FOR_DIFFUSE
    #define NON_LINEAR_ACCUM_SPEED      nonLinearAccumSpeed.x  // [注解] Diffuse 的累积速度
    #define ACCUM_SPEED                 data1.x                // [注解] Diffuse 的实际累积速度（来自 temporal pass）
    #define CHECKERBOARD                gDiffCheckerboard      // [注解] Diffuse 棋盘格模式（0=偶数, 1=奇数, 2=全分辨率）
    #define MIN_MATERIAL                gDiffMinMaterial       // [注解] Diffuse 材质 ID 匹配阈值
    #define ROUGHNESS                   1.0                    // [注解] Diffuse 等价粗糙度 = 1（完全漫反射）
    #define INPUT                       gIn_Diff
    #define INPUT_SH                    gIn_DiffSh             // [注解] SH (Spherical Harmonics) 模式的输入
    #define OUTPUT                      gOut_Diff
    #define OUTPUT_SH                   gOut_DiffSh
    #define OUTPUT_COPY                 gOut_DiffCopy          // [注解] PostBlur 时输出的副本（用于 temporal stabilization）
    #define OUTPUT_SH_COPY              gOut_DiffShCopy
#else
    #define USE_SCREEN_SPACE            REBLUR_USE_SCREEN_SPACE_SAMPLING_FOR_SPECULAR
    #define NON_LINEAR_ACCUM_SPEED      nonLinearAccumSpeed.y  // [注解] Specular 的累积速度
    #define ACCUM_SPEED                 data1.y
    #define CHECKERBOARD                gSpecCheckerboard
    #define MIN_MATERIAL                gSpecMinMaterial
    #define ROUGHNESS                   roughness              // [注解] 使用实际粗糙度
    #define INPUT                       gIn_Spec
    #define INPUT_SH                    gIn_SpecSh
    #define OUTPUT                      gOut_Spec
    #define OUTPUT_SH                   gOut_SpecSh
    #define OUTPUT_COPY                 gOut_SpecCopy
    #define OUTPUT_SH_COPY              gOut_SpecShCopy
#endif

// =====================================================================================
// [注解] ===================== 滤波主体开始（裸代码块 {}）=====================
//
// 注意：这里直接用 {} 包裹，没有函数声明
// 因为此代码被 #include 到调用方的函数体内部，共享调用方的局部变量：
//   pixelPos, pixelUv, viewZ, N, Nv, Xv, Vv, NoV, roughness,
//   frustumSize, rotator, nonLinearAccumSpeed, materialID,
//   checkerboard, checkerboardPos, wc (棋盘格相关)
// =====================================================================================
{
    // [注解] pos = 输入纹理的采样坐标
    uint2 pos = pixelPos;

    // [注解] Pre-Pass + 棋盘格模式下：输入纹理是半宽的
    //        CHECKERBOARD != 2 表示启用了棋盘格（2=全分辨率）
    //        pos.x >>= 1: x 坐标除以 2 来索引半宽纹理
#if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
    pos.x >>= CHECKERBOARD == 2 ? 0 : 1;
#endif

    // [注解] sum = 中心像素的权重（初始为 1.0）
    //        result = 中心像素的值（辐照度 + hit distance）
    float sum = 1.0;
    REBLUR_TYPE result = INPUT[ pos ];
    // [注解] REBLUR_TYPE 根据 NRD_MODE 不同而不同：
    //   正常模式: float4 (radiance.rgb + hitDist)
    //   Occlusion: float (标量 hit dist)

    // [注解] SH 模式：同时处理球谐系数
    #if( NRD_MODE == SH )
        REBLUR_SH_TYPE resultSh = INPUT_SH[ pos ];
    #endif

    // =====================================================================================
    // [注解] Pre-Pass 特有：棋盘格采样的初始处理
    //
    // 如果当前像素在本帧的棋盘格中 **没有被采样**（checkerboard != CHECKERBOARD），
    // 则将 sum 和 result 清零，完全依赖邻域来重建
    // =====================================================================================
#if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
    #if( NRD_SUPPORTS_CHECKERBOARD == 1 )
        if( CHECKERBOARD != 2 && checkerboard != CHECKERBOARD )
        {
            // [注解] 当前像素没有有效采样 → 清零，完全依赖空间滤波重建
            sum = 0;
            result = 0;
            #if( NRD_MODE == SH )
                resultSh = 0;
            #endif
        }
    #endif

    // [注解] 如果最大模糊半径为 0，跳过空间滤波（只做 checkerboard resolve）
    //        花括号不匹配是故意的 —— 这个 if 的结尾 } 在文件下方（与 checkerboard fallback 配对）
    if( MAX_BLUR_RADIUS != 0.0 )
    {
    // [注解] Specular Pre-Pass 初始化随机数生成器
    //        用于后续的随机采样决策（hit distance for tracking）
    #if( REBLUR_SPATIAL_LOBE == REBLUR_SPEC )
        Rng::Hash::Initialize( pixelPos, gFrameIndex );
    #endif
#endif

        // =====================================================================================
        // [注解] 缩放系数（A-trous 风格的多尺度行为）
        //
        // A-trous 是一种经典的多分辨率滤波方法，通过增大步长来扩大感受野
        // REBLUR 不直接用 A-trous，而是通过不同 pass 使用不同的缩放系数实现类似效果：
        //   - Pre-Pass:  小半径（radiusScale 小），轻量级预滤波
        //   - Blur:      中等半径，主要的空间降噪
        //   - Post-Blur: 大半径，最终精细化
        //
        // fractionScale: 控制权重衰减速度
        //   Pre-Pass 权重更宽松（fractionScale 大），Blur/PostBlur 更严格
        // =====================================================================================
    #if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
        float radiusScale = REBLUR_PRE_PASS_RADIUS_SCALE;
        float fractionScale = REBLUR_PRE_PASS_FRACTION_SCALE;
    #elif( REBLUR_SPATIAL_PASS == REBLUR_BLUR )
        float radiusScale = REBLUR_BLUR_RADIUS_SCALE;
        float fractionScale = REBLUR_BLUR_FRACTION_SCALE;
    #elif( REBLUR_SPATIAL_PASS == REBLUR_POST_BLUR )
        float radiusScale = REBLUR_POST_BLUR_RADIUS_SCALE;
        float fractionScale = REBLUR_POST_BLUR_FRACTION_SCALE;
    #endif

        // =====================================================================================
        // [注解] Hit Distance Factor（命中距离因子）
        //
        // 这是 REBLUR 自适应模糊半径的核心机制：
        //   hit distance 小 → 模糊半径小（反射/光照变化剧烈的区域需要保留细节）
        //   hit distance 大 → 模糊半径大（远处的间接光照变化平缓，可以更积极地模糊）
        // =====================================================================================

        // [注解] Dv: Specular 主方向（Dominant Direction）
        //        对于 specular 反射，光照主要来自一个特定方向（取决于粗糙度和观察角度）
        //        ROUGHNESS=1.0 时（diffuse），主方向退化为法线方向
        //        ML_SPECULAR_DOMINANT_DIRECTION_G2: GGX G2 项的近似
        float4 Dv = ImportanceSampling::GetSpecularDominantDirection( Nv, Vv, ROUGHNESS, ML_SPECULAR_DOMINANT_DIRECTION_G2 );

        // [注解] NoD: 法线与主方向的点积
        float NoD = abs( dot( Nv, Dv.xyz ) );

        // [注解] smc = Spec Magic Curve: 一个基于粗糙度的非线性映射曲线
        //        0.5 参数匹配之前使用的 sqrt(ROUGHNESS)
        //        修复了 roughness < 0.1 时的问题
        //        smc 用于调制 blur radius（低粗糙度 → smc 小 → 模糊半径小）
        float smc = GetSpecMagicCurve( ROUGHNESS, 0.5 );

        // [注解] 将归一化的 hit distance 转换为世界空间距离
        //        hitDistScale = 归一化因子（与 viewZ、粗糙度相关）
        float hitDistScale = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings.xyz, ROUGHNESS );
        float hitDist = ExtractHitDist( result ) * hitDistScale;

        // [注解] hitDistFactor ∈ [0, 1]: hit distance 相对于 frustumSize 的比例
        //        hitDist 很小 → factor 接近 0 → 小模糊半径
        //        hitDist 很大 → factor 接近 1 → 大模糊半径
        //        注释说明：用 hitDist * NoD 反而会恶化掠射角的降噪效果
        float hitDistFactor = GetHitDistFactor( hitDist, frustumSize );

        // =====================================================================================
        // [注解] 自适应模糊半径计算
        //
        // blurRadius 综合考虑多个因子：
        //   1. hitDistFactor: hit distance 越大 → 模糊越大
        //   2. NON_LINEAR_ACCUM_SPEED: 累积了更多历史帧 → 模糊可以更小
        //      （Pre-Pass 中不使用此因子，因为还没做 temporal）
        //   3. radiusScale: pass 级别的缩放系数
        //   4. smc (Spec Magic Curve): 粗糙度调制
        //   5. MAX_BLUR_RADIUS: 用户设置的最大半径上限
        //   6. gMinBlurRadius: 最小模糊半径下限
        // =====================================================================================
    #if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
        // [注解] Pre-Pass: 只考虑 hitDistFactor（没有 temporal 信息）
        float areaFactor = hitDistFactor;
    #else
        // [注解] Blur/PostBlur: hitDistFactor × 累积速度
        float areaFactor = hitDistFactor * NON_LINEAR_ACCUM_SPEED;
    #endif

        // [注解] 核心公式：
        //   blurRadius = radiusScale * sqrt(areaFactor)
        //   sqrt 是因为 areaFactor 影响的是面积（半径² ∝ 面积），取 sqrt 得到半径
        float blurRadius = radiusScale * Math::Sqrt01( areaFactor );

        // [注解] 限制在 [0, MAX_BLUR_RADIUS * smc] 范围内
        //   saturate(blurRadius) 确保 ∈ [0,1]
        //   乘以 MAX_BLUR_RADIUS 得到实际像素半径
        //   再乘以 smc 进行粗糙度调制
        blurRadius = saturate( blurRadius ) * MAX_BLUR_RADIUS * smc;

        // [注解] 确保不低于最小模糊半径（也经过 smc 调制）
        blurRadius = max( blurRadius, gMinBlurRadius * smc );

        // =====================================================================================
        // [注解] Specular Pre-Pass 特有："In-Lobe" 模糊限制
        //
        // 对于 specular pre-pass，MAX_BLUR_RADIUS 可能很大，但我们不想模糊超出
        // specular lobe 的范围（否则会引入 specular 泄漏 / 能量丢失）
        //
        // 计算方法：
        //   1. lobeTanHalfAngle: specular lobe 的半角正切值
        //      REBLUR_MAX_PERCENT_OF_LOBE_VOLUME_FOR_PRE_PASS: 使用 lobe 体积的百分比
        //   2. worldLobeRadius: lobe 在反射命中点处的世界空间半径
        //      = hitDist * NoD * tan(halfAngle)
        //   3. lobeRadius: 转换为屏幕空间像素半径
        //   4. blurRadius = min(blurRadius, lobeRadius): 不超过 lobe 范围
        // =====================================================================================
    #if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS && REBLUR_SPATIAL_LOBE == REBLUR_SPEC )
        float lobeTanHalfAngle = ImportanceSampling::GetSpecularLobeTanHalfAngle( ROUGHNESS, REBLUR_MAX_PERCENT_OF_LOBE_VOLUME_FOR_PRE_PASS );
        float worldLobeRadius = hitDist * NoD * lobeTanHalfAngle;

        // [注解] PixelRadiusToWorld 的逆操作：世界空间距离 → 屏幕像素数
        //        viewZ + hitDist * Dv.w: 反射命中点的近似深度
        //        Dv.w: 主方向的深度缩放因子
        float lobeRadius = worldLobeRadius / PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ + hitDist * Dv.w );

        blurRadius = min( blurRadius, lobeRadius );
    #endif

        // =====================================================================================
        // [注解] 各种空间权重参数的预计算
        // =====================================================================================

        // [注解] 几何平面距离权重参数（与 HitDistReconstruction 相同的严格权重）
        float2 geometryWeightParams = GetGeometryWeightParams( gPlaneDistSensitivity, frustumSize, Xv, Nv );

        // [注解] 法线权重参数
        //        NON_LINEAR_ACCUM_SPEED: 影响权重严格程度（累积越多 → 越严格，减少模糊）
        //        gLobeAngleFraction: lobe 角度的分数（全局配置）
        //        fractionScale: pass 级别的缩放
        float normalWeightParam = GetNormalWeightParam( NON_LINEAR_ACCUM_SPEED, gLobeAngleFraction, ROUGHNESS ) / fractionScale;

        // [注解] 粗糙度权重参数（用于 specular 通道）
        float2 roughnessWeightParams = GetRoughnessWeightParams( ROUGHNESS, gRoughnessFraction * fractionScale );

        // [注解] Hit distance 权重参数
        //        用于判断邻域的 hit distance 与中心是否一致
        //        差异大 → 权重低（可能来自不同的反射路径）
        float2 hitDistanceWeightParams = GetHitDistanceWeightParams( ExtractHitDist( result ), NON_LINEAR_ACCUM_SPEED );

        // [注解] 最小 hit distance 权重
        //        确保即使 hit distance 差异较大，样本仍有一定的最小贡献
        //        smc 调制使低粗糙度时最小权重更小（镜面反射需要更精确的匹配）
        float minHitDistWeight = gMinHitDistanceWeight * fractionScale * smc;

        // [注解] 非 Pre-Pass + 非 Occlusion 模式下：
        //        进一步用 NON_LINEAR_ACCUM_SPEED 调低 minHitDistWeight
        //        目的是 "挤出更多阴影细节"（squeeze more shadow details）
        //        累积越多帧 → minHitDistWeight 越小 → hit distance 权重更严格 → 阴影更锐利
    #if( REBLUR_SPATIAL_PASS != REBLUR_PRE_PASS && NRD_MODE != OCCLUSION && NRD_MODE != DO )
        minHitDistWeight *= NON_LINEAR_ACCUM_SPEED;
    #endif

        // =====================================================================================
        // [注解] 采样空间设置：屏幕空间 vs 世界空间
        //
        // Pre-Pass 总是使用 **屏幕空间采样**（效果更均匀，能更好地抑制 residual boiling）
        // Blur/PostBlur 可选择屏幕空间或世界空间：
        //   - 屏幕空间：采样点在 UV 空间均匀分布，通过旋转 Poisson 盘实现
        //   - 世界空间：采样点在 3D 切平面上分布，需要构建切线-副切线基（TvBv）
        //     世界空间采样能更好地适应几何形状，但计算量更大
        // =====================================================================================
    #if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS || USE_SCREEN_SPACE == 1 )
        // ---- 屏幕空间采样设置 ----

        #if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
            // [注解] Pre-Pass: skew = 1（均匀采样，不做各向异性拉伸）
            //        注释说明均匀屏幕空间采样能更好地抑制 residual boiling
            float2 skew = 1.0;
        #else
            #if( REBLUR_SPATIAL_LOBE == REBLUR_DIFF )
                // [注解] Diffuse Blur/PostBlur: 基于法线方向做各向异性拉伸
                //        在法线几乎平行于屏幕的方向（|Nv.x| 或 |Nv.y| 大）缩小采样范围
                //        在法线垂直于屏幕的方向保持不变
                //        NoV 插值使掠射角时拉伸减弱
                float2 skew = lerp( 1.0 - abs( Nv.xy ), 1.0, NoV );
                skew /= max( skew.x, skew.y );  // [注解] 归一化使较大分量为 1
            #else
                // [注解] Specular: 目前不做拉伸（TODO: 未来可能改进）
                float2 skew = 1.0;
            #endif
        #endif

        // [注解] skew 乘以 gRectSizeInv * blurRadius 得到 UV 空间的采样步长
        //        然后通过 ScaleRotator 将旋转矩阵缩放到正确的采样范围
        skew *= gRectSizeInv * blurRadius;
        float4 scaledRotator = Geometry::ScaleRotator( rotator, skew );
    #else
        // ---- 世界空间采样设置 ----

        #if( REBLUR_SPATIAL_LOBE == REBLUR_DIFF )
            // [注解] Diffuse: 无各向异性拉伸，采样方向沿法线方向
            float skewFactor = 1.0;
            float3 bentDv = Nv;  // [注解] bent direction = 法线方向
        #else
            // [注解] Specular: 沿 specular dominant direction 弯曲采样核
            //        bentFactor: 基于 hitDistFactor 的弯曲程度
            //        hitDistFactor 大 → 反射远处物体 → 采样核更偏向 dominant direction
            //        hitDistFactor 小 → 反射近处物体 → 采样核偏向法线方向
            float bentFactor = sqrt( hitDistFactor );

            // [注解] skewFactor: 采样核的各向异性拉伸因子
            //        粗糙度高 → 拉伸少（接近 1）；粗糙度低 → 拉伸多（接近 0.25）
            //        NoD 大 → 拉伸少；NoD 小 → 拉伸多
            //        NON_LINEAR_ACCUM_SPEED 大 → 拉伸少（新区域保守一些）
            //        bentFactor 小 → 拉伸少（近距离反射不需要过度拉伸）
            float skewFactor = lerp( 0.25 + 0.75 * ROUGHNESS, 1.0, NoD );
            skewFactor = lerp( skewFactor, 1.0, NON_LINEAR_ACCUM_SPEED );
            skewFactor = lerp( 1.0, skewFactor, bentFactor );

            // [注解] bentDv: 弯曲的采样方向（在法线和 dominant direction 之间插值）
            float3 bentDv = normalize( lerp( Nv.xyz, Dv.xyz, bentFactor ) );
        #endif

        // [注解] 将像素空间的 blur radius 转换为世界空间半径
        float worldRadius = PixelRadiusToWorld( gUnproject, gOrthoMode, blurRadius, viewZ );

        // [注解] 构建采样核基（Kernel Basis）：两个正交的切线向量
        //        TvBv[0] = T (切线方向)，TvBv[1] = B (副切线方向)
        //        基于 bentDv 和 Nv 构建
        float2x3 TvBv = GetKernelBasis( bentDv, Nv );

        // [注解] 用世界空间半径和 skewFactor 缩放基向量
        //        T 方向乘以 skewFactor（可能被拉伸/压缩）
        //        B 方向除以 skewFactor（保持面积不变的各向异性拉伸）
        TvBv[ 0 ] *= worldRadius * skewFactor;
        TvBv[ 1 ] *= worldRadius / skewFactor;
    #endif

        // =====================================================================================
        // [注解] ===================== 采样循环（核心计算）=====================
        // =====================================================================================

        // [注解] hitDistForTracking: 用于 specular 运动估计的最小 hit distance
        //        初始化为 NRD_INF（如果中心 hit distance 为 0，即无效采样）
        //        或者为 hitDist（中心有效时）
        float hitDistForTracking = hitDist == 0.0 ? NRD_INF : hitDist;

        [unroll]  // [注解] 循环展开优化
        for( uint n = 0; n < POISSON_SAMPLE_NUM; n++ )
        {
            // [注解] Poisson 采样点：offset.xy = 2D 偏移，offset.z = 归一化距离（用于高斯权重）
            float3 offset = POISSON_SAMPLES( n );

            // =====================================================================================
            // [注解] 计算采样坐标
            // =====================================================================================
        #if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS || USE_SCREEN_SPACE == 1 )
            // [注解] 屏幕空间采样：
            //        将 Poisson 偏移旋转后加到中心 UV 上
            //        scaledRotator 包含了旋转 + 缩放（UV 空间的采样步长）
            float2 uv = pixelUv + Geometry::RotateVector( scaledRotator, offset.xy );
        #else
            // [注解] 世界空间采样：
            //        将 Poisson 偏移投影到 3D 切平面上，再投影回屏幕空间
            //        使用 gViewToClip 矩阵做投影
            float2 uv = GetKernelSampleCoordinates( gViewToClip, offset, Xv, TvBv[ 0 ], TvBv[ 1 ], rotator );
        #endif

            // [注解] UV 镜像处理：如果采样点超出屏幕范围 [0,1]，将其镜像回来
            //        这样不会浪费采样点（否则超出屏幕的采样就白费了）
            //        镜像后 offset.z 不再有效，所以 w 设为 1.0
            float2 uv01 = saturate( uv );
            float w = GetGaussianWeight( offset.z );
            if( any( uv != uv01 ) )
            {
                uv = MirrorUv( uv );
                w = 1.0;
            }

            // [注解] UV → 像素坐标（uv 不会等于 1.0，所以 floor 是安全的）
            int2 pos = uv * gRectSize;

            // =====================================================================================
            // [注解] 棋盘格模式下的像素位置调整
            //
            // 在棋盘格渲染中，只有一半像素有有效采样
            // 如果采样到了无效像素，需要左/右偏移 1 个像素到最近的有效像素
            // 交替使用 -1 和 +1 偏移（n & 0x1）来避免系统性偏差
            // =====================================================================================
            int checkerboardX = pos.x;
            #if( NRD_SUPPORTS_CHECKERBOARD == 1 && REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
                if( CHECKERBOARD != 2 )
                {
                    // [注解] 如果采样到的像素不是当前帧的有效棋盘格像素，偏移到最近的有效像素
                    int shift = ( ( n & 0x1 ) == 0 ) ? -1 : 1;
                    pos.x += Sequence::CheckerBoard( pos, gFrameIndex ) != CHECKERBOARD ? shift : 0;
                    checkerboardX = pos.x >> 1;  // [注解] 半宽纹理索引

                    // [注解] 偏移后可能超出屏幕范围，此时权重设为 0
                    w = pos.x < 0.0 || pos.x > gRectSizeMinusOne.x ? 0.0 : w;
                }
            #endif

            // =====================================================================================
            // [注解] 获取采样点数据
            // =====================================================================================

            // [注解] 读取采样点深度
            //        PostBlur 时不加 RectOrigin 偏移（因为输入已经是 rect 空间）
        #if( REBLUR_SPATIAL_PASS == REBLUR_POST_BLUR )
            float zs = UnpackViewZ( gIn_ViewZ[ pos ] );
        #else
            float zs = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pos ) ] );
        #endif

            // [注解] 重建采样点的观察空间 3D 位置
            float3 Xvs = Geometry::ReconstructViewPosition( float2( pos + 0.5 ) * gRectSizeInv, gFrustum, zs, gOrthoMode );

            // [注解] 读取并解包采样点的法线、粗糙度、材质 ID
            float materialIDs;
            float4 Ns = gIn_Normal_Roughness[ WithRectOrigin( pos ) ];
            Ns = NRD_FrontEnd_UnpackNormalAndRoughness( Ns, materialIDs );

            // =====================================================================================
            // [注解] 综合权重计算
            // =====================================================================================

            // [注解] 1. 法线夹角（用于法线权重）
            float angle = Math::AcosApprox( dot( N, Ns.xyz ) );

            // [注解] 2. 采样点在中心法线平面上的投影（用于几何权重）
            float NoX = dot( Nv, Xvs );

            // [注解] 3. 几何平面距离权重（严格截断 —— 防止跨表面混合）
            w *= ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );

            // [注解] 4. 材质 ID 权重：不同材质的像素不应互相模糊
            //        CompareMaterials: 如果 materialID 匹配返回 1，否则返回 0 或很小的值
            //        MIN_MATERIAL: 最小材质匹配阈值
            w *= CompareMaterials( materialID, materialIDs, MIN_MATERIAL );

            // [注解] 5. 法线权重（指数衰减）
            w *= ComputeWeight( angle, normalWeightParam, 0.0 );

            // [注解] 6. 粗糙度权重（仅 specular 通道）
        #if( REBLUR_SPATIAL_LOBE == REBLUR_SPEC )
            w *= ComputeWeight( Ns.w, roughnessWeightParams.x, roughnessWeightParams.y );
        #endif

            // [注解] 7. 深度有效性：超出降噪范围的采样点权重设为 0
            w = zs < gDenoisingRange ? w : 0.0;

            // [注解] 读取采样点的辐照度 + hit distance
            REBLUR_TYPE s = INPUT[ int2( checkerboardX, pos.y ) ];
            s = Denanify( w, s );  // [注解] NaN 防护

            // =====================================================================================
            // [注解] Specular Pre-Pass 特有逻辑
            // =====================================================================================
        #if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS && REBLUR_SPATIAL_LOBE == REBLUR_SPEC )
            // [注解] ==== Hit Distance for Tracking ====
            //
            // 在邻域中寻找最小的有效 hit distance，用于后续 temporal reprojection 的运动估计
            // 最小 hit distance 对应最近的反射命中点，是运动估计最可靠的参考
            //
            // 使用随机概率采样（Rng::Hash::GetFloat() < geometryWeight）而非确定性比较
            // 这避免了每次都选择同一个最近点，增加了鲁棒性
            //
            // 注释中的 TODO：trimming 关闭时，最小 hitDist 可能来自概率很低的采样，
            // 会恶化 reprojection 质量
            float hs = ExtractHitDist( s ) * _REBLUR_GetHitDistanceNormalization( zs, gHitDistSettings.xyz, Ns.w );
            float geometryWeight = w * NoV * float( hs != 0.0 );
            if( Rng::Hash::GetFloat( ) < geometryWeight )
                hitDistForTracking = min( hitDistForTracking, hs );

            // [注解] ==== 稀疏高亮样本问题 ====
            //
            // 注释原文：在极少数情况下，高亮样本非常稀疏，pre-pass 会把一个孤立的亮像素
            // 扩展成一个亮 blob，反而恶化了情况。
            // 解决方案：用 gUsePrepassNotOnlyForSpecularMotionEstimation 开关
            // 如果设为 0，pre-pass 只做 hit distance for tracking，不做实际的空间模糊
            // TODO: 是否有更好的解决方案？
            w *= gUsePrepassNotOnlyForSpecularMotionEstimation;

            // [注解] ==== 反射接触点距离权重 ====
            //
            // 如果邻域采样的反射命中点非常靠近反射表面（hs 很小），
            // 说明是 "contact shadow" 或近距离反射，不应被模糊掉
            //
            // d = 中心像素与采样点的 3D 距离（观察空间）
            // t = hs / (d + hitDist): 比值越小 → 反射命中点越近 → 权重越低
            // 高粗糙度时放松此限制（roughness > 0.5 时 t 的影响减弱）
            float d = length( Xvs - Xv ) + NRD_EPS;
            float t = hs / ( d + hitDist );
            w *= lerp( saturate( t ), 1.0, Math::LinearStep( 0.5, 1.0, ROUGHNESS ) );
        #endif

            // [注解] 8. Hit distance 权重 + 最小权重保底
            //        ComputeExponentialWeight: 基于 hit distance 差异的指数衰减
            //        minHitDistWeight: 保底权重，确保即使 hit distance 差异大也有一定贡献
            w *= minHitDistWeight + ComputeExponentialWeight( ExtractHitDist( s ), hitDistanceWeightParams.x, hitDistanceWeightParams.y );

            // =====================================================================================
            // [注解] 累加加权结果
            // =====================================================================================
            sum += w;

            result += s * w;
            #if( NRD_MODE == SH )
                // [注解] SH 模式下同时累加球谐系数
                REBLUR_SH_TYPE sh = INPUT_SH[ int2( checkerboardX, pos.y ) ];
                sh = Denanify( w, sh );
                resultSh += sh * w;
            #endif
        }

        // =====================================================================================
        // [注解] 归一化（除以总权重得到加权平均）
        // =====================================================================================
        float invSum = Math::PositiveRcp( sum );
        result *= invSum;
        #if( NRD_MODE == SH )
            resultSh *= invSum;
        #endif

        // =====================================================================================
        // [注解] 非 Pre-Pass 模式下：保持原始 hit distance 不变
        //
        // 关键原理："Keep hit distances unprocessed to avoid bias and self-inference"
        //   在 Blur/PostBlur 中，hit distance 不参与空间平均
        //   如果把 hit distance 也模糊了，后续 pass 使用模糊后的 hit distance
        //   来计算权重（self-inference），会产生偏差
        //   所以将 .w 恢复为原始值（hitDist / hitDistScale = 归一化的原始 hit distance）
        //
        // Pre-Pass 不受此限制（因为 pre-pass 的结果不会被自身引用）
        // Occlusion 模式也不受影响（只有标量）
        // =====================================================================================
    #if( REBLUR_SPATIAL_PASS != REBLUR_PRE_PASS && NRD_MODE != OCCLUSION && NRD_MODE != DO )
        result.w = hitDist / hitDistScale;
    #endif

// =====================================================================================
// [注解] Pre-Pass 特有的收尾逻辑
// =====================================================================================
#if( REBLUR_SPATIAL_PASS == REBLUR_PRE_PASS )
    #if( REBLUR_SPATIAL_LOBE == REBLUR_SPEC )
        // [注解] 输出 specular hit distance for tracking
        //        如果始终为 NRD_INF（没找到有效样本），输出 0
        //        此值被后续 temporal reprojection pass 使用，用于估计反射表面的运动
        gOut_SpecHitDistForTracking[ pixelPos ] = hitDistForTracking == NRD_INF ? 0.0 : hitDistForTracking;
    #endif
    }
    // [注解] 上面的 } 匹配 "if( MAX_BLUR_RADIUS != 0.0 )" 的 {
    //        如果 MAX_BLUR_RADIUS == 0，跳过所有空间滤波，直接到这里

    // =====================================================================================
    // [注解] 棋盘格 Fallback（Pre-Pass + 棋盘格模式）
    //
    // 如果空间滤波完全失败（sum == 0，即没有任何有效样本），
    // 使用简单的左右邻居加权平均来恢复棋盘格缺失的像素
    // wc 是之前预计算的棋盘格恢复权重
    // =====================================================================================
    #if( NRD_SUPPORTS_CHECKERBOARD == 1 )
        [branch]  // [注解] 动态分支（大多数像素不会进入此分支）
        if( sum == 0.0 )
        {
            // [注解] 从左右邻居读取数据
            REBLUR_TYPE s0 = INPUT[ checkerboardPos.xz ];
            REBLUR_TYPE s1 = INPUT[ checkerboardPos.yz ];

            // [注解] NaN 防护
            s0 = Denanify( wc.x, s0 );
            s1 = Denanify( wc.y, s1 );

            // [注解] 加权平均
            result = s0 * wc.x + s1 * wc.y;

            #if( NRD_MODE == SH )
                REBLUR_SH_TYPE sh0 = INPUT_SH[ checkerboardPos.xz ];
                REBLUR_SH_TYPE sh1 = INPUT_SH[ checkerboardPos.yz ];

                sh0 = Denanify( wc.x, sh0 );
                sh1 = Denanify( wc.y, sh1 );

                resultSh = sh0 * wc.x + sh1 * wc.y;
            #endif
        }
    #endif
#endif

    // =====================================================================================
    // [注解] 写入输出
    // =====================================================================================
    OUTPUT[ pixelPos ] = result;
    #if( NRD_MODE == SH )
        OUTPUT_SH[ pixelPos ] = resultSh;
    #endif

    // =====================================================================================
    // [注解] PostBlur 特有：输出副本（用于 Temporal Stabilization）
    //
    // 当 TEMPORAL_STABILIZATION == 0 时（不使用 temporal stabilization pass）
    // PostBlur 需要直接输出最终结果的副本
    //
    // 如果 gReturnHistoryLengthInsteadOfOcclusion == true:
    //   将 .w 替换为累积速度（ACCUM_SPEED），用于调试或特殊用途
    // =====================================================================================
#if( REBLUR_SPATIAL_PASS == REBLUR_POST_BLUR && TEMPORAL_STABILIZATION == 0 )
    #if( NRD_MODE != OCCLUSION && NRD_MODE != DO )
        result.w = gReturnHistoryLengthInsteadOfOcclusion ? ACCUM_SPEED : result.w;
    #endif

    OUTPUT_COPY[ pixelPos ] = result;
    #if( NRD_MODE == SH )
        OUTPUT_SH_COPY[ pixelPos ] = resultSh;
    #endif
#endif
}

// =====================================================================================
// [注解] 宏清理（#undef）
//
// 清除所有在此文件中定义的宏，以便此文件可以被再次 #include（使用不同的参数）
// 例如：第一次 include 处理 Diffuse，第二次 include 处理 Specular
//
// 注意：REBLUR_SPATIAL_PASS 不在这里 undef，因为它在整个 CS 中保持不变
//       只有 REBLUR_SPATIAL_LOBE 和 MAX_BLUR_RADIUS 需要在两次 include 之间切换
// =====================================================================================
#undef POISSON_SAMPLE_NUM
#undef POISSON_SAMPLES
#undef USE_SCREEN_SPACE
#undef NON_LINEAR_ACCUM_SPEED
#undef ACCUM_SPEED
#undef CHECKERBOARD
#undef MIN_MATERIAL
#undef ROUGHNESS
#undef INPUT
#undef INPUT_SH
#undef OUTPUT
#undef OUTPUT_SH
#undef OUTPUT_COPY
#undef OUTPUT_SH_COPY

#undef REBLUR_SPATIAL_LOBE
#undef MAX_BLUR_RADIUS
