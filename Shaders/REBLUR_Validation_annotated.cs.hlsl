/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR Validation（调试可视化 pass）
//
// 它不是 REBLUR 的降噪主 pass，而是一个 **开发/验证用的可视化面板**：
//   - 将输入的 normal / roughness / viewZ / motion vectors / history data 可视化
//   - 帮助开发者快速检查前端输入是否正确、内部 history 是否稳定、hit distance 是否异常
//   - 输出是一个 4x4 的调试拼图，每个 viewport 显示一种诊断结果
//
// 这类 pass 的设计思路很典型：
//   1. 同一张输出纹理被切成 16 个小窗口
//   2. 每个小窗口复用同一套输入数据，但用不同方式着色
//   3. 叠加文字标签，方便运行时直接识别当前在看什么
//
// 注意：
//   - 并不是 16 个格子都实现了内容，未使用的格子会保持先前结果（初始通常为 0）
//   - 这个 pass 依赖的是“已经存在的中间纹理”，不会改变 REBLUR 主流程的行为
// =====================================================================================

// [注解] NRD 基础宏、导出宏、公共类型
#include "NRD.hlsli"

// [注解] 数学工具库
#include "ml.hlsli"

// [注解] REBLUR 配置宏
#include "REBLUR_Config.hlsli"

// [注解] Validation pass 的资源绑定：
//        输入包括法线/粗糙度、viewZ、MV、Data1、Data2、diff、spec
//        输出只有一张调试纹理 `gOut_Validation`
#include "REBLUR_Validation.resources.hlsli"

// [注解] 通用几何、颜色、文字绘制、坐标工具
#include "Common.hlsli"

// [注解] 这里强制把 `NRD_SPEC` 设为 1，原因不是“此 pass 只处理 specular”，
//        而是为了复用 `REBLUR_Common.hlsli` 里依赖 spec 宏分支的辅助函数，
//        特别是 `UnpackData2`。
#undef NRD_SPEC
#define NRD_SPEC 1 // see "UnpackData2"

// [注解] REBLUR 公共函数：数据打包/解包、hit distance 提取等
#include "REBLUR_Common.hlsli"

// [注解] 每个调试小窗口占整张输出的 1/4 宽和 1/4 高，因此总共是 4x4 = 16 格。
#define VIEWPORT_SIZE   0.25

// [注解] 绘制标题文字时的像素偏移，避免文字贴边
#define OFFSET          5

/*
[注解] 4x4 viewport 布局如下：
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15

当前文件中真正有内容的格子主要是：
 0  = NORMALS
 1  = ROUGHNESS
 2  = Z
 3  = MV
 4  = UNITS / JITTER / ROTATORS
 7  = VIRTUAL HISTORY（仅 spec）
 8  = DIFF FRAMES（仅 diffuse）
11  = SPEC FRAMES（仅 spec）
12 = DIFF HITT（仅 diffuse）
15 = SPEC HITT（仅 spec）
*/

[numthreads( 16, 16, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    // [注解] 默认的 CTA 排列宏，负责生成 `pixelPos`
    NRD_CTA_ORDER_DEFAULT;

    // [注解] 如果整条 history 被重置，则直接把调试输出清零。
    //        这是因为某些格子会跨帧累积显示（例如 rotators 区域），
    //        reset 时必须主动清空旧内容。
    if( gResetHistory != 0 )
    {
        gOut_Validation[ pixelPos ] = 0;
        return;
    }

    // [注解] 当前输出像素在整张 validation 纹理中的归一化坐标
    float2 pixelUv = float2( pixelPos + 0.5 ) / gResourceSize;

    // [注解] `viewportUv`：当前像素在“所属小窗口内部”的局部 UV，范围约为 [0, 1)
    //        `viewportId`：当前像素落在哪个小窗口（0~3, 0~3）
    //        `viewportIndex`：把 2D 小窗口索引压成 0~15 的一维编号
    float2 viewportUv = frac( pixelUv / VIEWPORT_SIZE );
    float2 viewportId = floor( pixelUv / VIEWPORT_SIZE );
    float viewportIndex = viewportId.y / VIEWPORT_SIZE + viewportId.x;

    // [注解] Validation pass 读取的是“被调试的实际输入分辨率”上的纹理。
    //        如果启用了动态分辨率/分辨率缩放，需要先把局部 viewport UV 映射回真实输入 UV。
    float2 viewportUvScaled = viewportUv * gResolutionScale;

    // [注解] 从输入纹理中取出当前要可视化的数据。
    //        这里都用 SampleLevel(..., 0) 进行显式 LOD=0 采样，避免 mip 干扰调试结果。
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness.SampleLevel( gNearestClamp, WithRectOffset( viewportUvScaled ), 0 ) );
    float viewZ = UnpackViewZ( gIn_ViewZ.SampleLevel( gNearestClamp, WithRectOffset( viewportUvScaled ), 0 ) );
    float3 mv = gIn_Mv.SampleLevel( gNearestClamp, WithRectOffset( viewportUvScaled ), 0 ) * gMvScale.xyz;

    // [注解] Diff / Spec 可能来自 checkerboard 纹理。
    //        当 checkerboard 打开时，横向分辨率只有一半，因此采样 UV 的 x 要缩半。
    float4 diff = gIn_Diff.SampleLevel( gNearestClamp, viewportUvScaled * float2( gDiffCheckerboard != 2 ? 0.5 : 1.0, 1.0 ), 0 );
    float4 spec = gIn_Spec.SampleLevel( gNearestClamp, viewportUvScaled * float2( gSpecCheckerboard != 2 ? 0.5 : 1.0, 1.0 ), 0 );

    // [注解] `Data1` 里保存的是 diffuse/spec 的历史帧数等信息，通常是归一化存储。
    //        这里乘回 `REBLUR_MAX_ACCUM_FRAME_NUM`，得到“接近真实含义”的帧数尺度，
    //        方便直接做可视化。
    // See "UnpackData1"
    REBLUR_DATA1_TYPE data1 = gIn_Data1.SampleLevel( gNearestClamp, viewportUvScaled, 0 );
    if( !gHasDiffuse )
        data1.y = data1.x;
    data1 *= REBLUR_MAX_ACCUM_FRAME_NUM;

    // [注解] `Data2` 存的是额外的 bitfield / history 元信息。
    //        `UnpackData2` 会解码出：
    //          - `data2.x`：这里被用于显示 virtual history 相关值
    //          - `bits`：紧凑编码的一组 flags
    //          - `smbAllowCatRom`：surface motion reprojection 时是否允许 CatRom
    //        Validation pass 并不会把这些全部画出来，只会拿其中一部分做 debug 展示。
    uint bits;
    bool smbAllowCatRom;
    float2 data2 = UnpackData2( gIn_Data2[ uint2( viewportUvScaled * gResourceSize ) ], bits, smbAllowCatRom );

    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    // [注解] 重建当前点的世界空间位置，用于：
    //   - 检查 motion vector 是否和“几何学上的上一帧投影位置”一致
    //   - 显示 world-space 单位格/抖动/rotator 调试图
    float3 Xv = Geometry::ReconstructViewPosition( viewportUv, gFrustum, abs( viewZ ), gOrthoMode );
    float3 X = Geometry::RotateVector( gViewToWorld, Xv );

    // [注解] 超出降噪范围的点在很多视图里会被特殊着色或直接抑制显示
    bool isInf = abs( viewZ ) > gDenoisingRange;

    // [注解] 这里的 checkerboard 不是“真实输入是否 checkerboard”的唯一来源，
    //        而是给 frames 可视化面板用的一个局部棋盘模式，用来标示特殊的低历史区域。
    bool checkerboard = Sequence::CheckerBoard( pixelPos >> 2, 0 );

    // [注解] 文本系统初始化：
    //   - 当前像素坐标
    //   - 当前小窗口左上角 + OFFSET 作为文字起点
    //   - 缩放系数为 1
    uint4 textState = Text::Init( pixelPos, viewportId * gResourceSize * VIEWPORT_SIZE + OFFSET, 1 );

    // [注解] 默认先取旧结果。
    //        绝大多数格子会被当前分支覆盖；
    //        但某些调试图（如 rotator 可视化）会利用旧值做跨帧累积展示。
    float4 result = gOut_Validation[ pixelPos ];

    // =====================================================================================
    // [注解] Viewport 0: World-space normal
    //
    // 经典法线可视化：[-1,1] -> [0,1]，R/G/B 对应 XYZ。
    // 如果法线方向明显错乱、颠倒、量化异常，这里会非常容易看出来。
    // =====================================================================================
    if( viewportIndex == 0 )
    {
        Text::Print_ch( 'N', textState );
        Text::Print_ch( 'O', textState );
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'M', textState );
        Text::Print_ch( 'A', textState );
        Text::Print_ch( 'L', textState );
        Text::Print_ch( 'S', textState );
        Text::Print_ch( '-', textState );
        Text::NextChar( textState );
        Text::Print_ui( NRD_NORMAL_ENCODING, textState );

        result.xyz = N * 0.5 + 0.5;
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 1: Linear roughness
    //
    // 直接把 roughness 作为灰度值输出。
    // 如果 roughness 贴图、编码或前端打包有问题，这一格会立刻暴露。
    // =====================================================================================
    else if( viewportIndex == 1 )
    {
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'O', textState );
        Text::Print_ch( 'U', textState );
        Text::Print_ch( 'G', textState );
        Text::Print_ch( 'H', textState );
        Text::Print_ch( 'N', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'S', textState );
        Text::Print_ch( 'S', textState );
        Text::Print_ch( '-', textState );
        Text::NextChar( textState );
        Text::Print_ui( NRD_ROUGHNESS_ENCODING, textState );

        result.xyz = roughness;
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 2: View Z
    //
    // 规则：
    //   - 蓝色：viewZ < 0
    //   - 绿色：viewZ > 0
    //   - 红色：超出 denoising range（当作 infinity / 无效降噪区）
    // 亮度通过一个压缩曲线映射，避免远距离深度全挤在一起。
    // =====================================================================================
    else if( viewportIndex == 2 )
    {
        Text::Print_ch( 'Z', textState );
        if( viewZ < 0 )
            Text::Print_ch( Text::Char_Minus, textState );

        float f = 0.1 * abs( viewZ ) / ( 1.0 + 0.1 * abs( viewZ ) ); // TODO: tuned for meters
        float3 color = viewZ < 0.0 ? float3( 0, 0, 1 ) : float3( 0, 1, 0 );

        result.xyz = isInf ? float3( 1, 0, 0 ) : color * f;
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 3: Motion Vector validation
    //
    // 核心思想：
    //   1. 用当前像素世界坐标 X 经过上一帧矩阵投影，得到“理论上的上一帧屏幕位置”
    //   2. 再用输入 motion vector 算出“MV 声称的上一帧位置”
    //   3. 两者做差
    //
    // 输出规则：
    //   - R/G：误差绝对值（单位约为像素）
    //   - B=1：MV 指向了屏幕外
    //
    // 这是检查 MV 是否缩放正确、坐标系是否一致、是否把世界空间 MV 当屏幕空间 MV 的好工具。
    // =====================================================================================
    else if( viewportIndex == 3 )
    {
        Text::Print_ch( 'M', textState );
        Text::Print_ch( 'V', textState );

        float2 viewportUvPrevExpected = Geometry::GetScreenUv( gWorldToClipPrev, X );

        float2 viewportUvPrev = viewportUv + mv.xy;
        if( gMvScale.w != 0.0 )
            viewportUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, X + mv );

        float2 uvDelta = ( viewportUvPrev - viewportUvPrevExpected ) * gRectSize;

        result.xyz = IsInScreenNearest( viewportUvPrev ) ? float3( abs( uvDelta ), 0 ) : float3( 0, 0, 1 );
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 4: World units / jitter / rotators 综合面板
    //
    // 这个窗口内部又被分成几个子区域：
    //   - 主区域：显示世界坐标的小数部分，相当于单位格纹理，可观察空间稳定性
    //   - 右上局部：显示 jitter 位置
    //   - 右半另一块：显示 pre / current / post rotator 的采样点分布
    //
    // 这是一个非常“开发者内部”的面板，用来确认：
    //   - TAA jitter 是否正确落在像素域内
    //   - 不同 rotator 是否按预期旋转 Poisson / special pattern
    //   - 世界空间重建是否在相机运动时平滑连续
    // =====================================================================================
    else if( viewportIndex == 4 )
    {
        Text::Print_ch( 'U', textState );
        Text::Print_ch( 'N', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'T', textState );
        Text::Print_ch( 'S', textState );

        float2 dim = float2( 0.5 * gResourceSize.y / gResourceSize.x, 0.5 );
        float2 dimInPixels = gResourceSize * VIEWPORT_SIZE * dim;

        float2 remappedUv = ( viewportUv - ( 1.0 - dim ) ) / dim;
        float2 remappedUv2 = ( viewportUv - float2( 1.0 - dim.x, 0.0 ) ) / dim;

        if( all( remappedUv > 0.0 ) )
        {
            // [注解] 右下小块：显示 jitter 在像素域中的位置。
            //        有效 jitter 用灰色点；若 jitter 超出 [0,1]，则用红色警告。
            float2 uv = gJitter + 0.5;
            bool isValid = all( saturate( uv ) == uv );
            int2 a = int2( saturate( uv ) * dimInPixels );
            int2 b = int2( remappedUv * dimInPixels );

            if( all( abs( a - b ) <= 1 ) && isValid )
                result.xyz = 0.66;

            if( all( abs( a - b ) <= 3 ) && !isValid )
                result.xyz = float3( 1.0, 0.0, 0.0 );
        }
        else if( all( remappedUv2 > 0.0 ) )
        {
            // [注解] 右侧另一块：显示三组 rotator（pre / current / post）后的采样分布。
            //        RGB 分别累计三种 rotator 的命中次数，所以最终颜色代表三套旋转模式的覆盖关系。
            float scale = 0.5;
            //scale *= sqrt( 1.0 / ( 1.0 + ( gFrameIndex % 16 ) ) );
            scale *= float( Math::ReverseBits4( gFrameIndex ) ) / 16.0;

            float4 rotatorPre = gRotatorPre;
            float4 rotator = gRotator;
            float4 rotatorPost = gRotatorPost;

            int2 b = int2( remappedUv2 * dimInPixels );

            [unroll]
            for( uint n = 0; n < 8; n++ )
            {
                float3 offset = g_Special8[ n ];
                offset *= scale;

                {
                    float2 uv = 0.5 + Geometry::RotateVector( rotatorPre, offset.xy );
                    int2 a = int2( saturate( uv ) * dimInPixels );

                    result.x += all( abs( a - b ) <= 1 );
                }

                {
                    float2 uv = 0.5 + Geometry::RotateVector( rotator, offset.xy );
                    int2 a = int2( saturate( uv ) * dimInPixels );

                    result.y += all( abs( a - b ) <= 1 );
                }

                {
                    float2 uv = 0.5 + Geometry::RotateVector( rotatorPost, offset.xy );
                    int2 a = int2( saturate( uv ) * dimInPixels );

                    result.z += all( abs( a - b ) <= 1 );
                }
            }

            // [注解] 每 256 帧清空一次，避免无限累加后整块饱和成纯白。
            result = gFrameIndex % 256 == 0 ? 0 : saturate( result );
        }
        else
        {
            // [注解] 主区域：显示世界坐标的小数部分 frac(X)。
            //        如果世界空间重建稳定，你会看到连续、平滑的彩色单位格图样。
            float roundingErrorCorrection = abs( viewZ ) * 0.001;

            result.xyz = frac( X + roundingErrorCorrection ) * float( !isInf );
        }

        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 7: Virtual history（仅 specular）
    //
    // 这里直接把 `data2.x` 当灰度输出。它反映的是 specular 路径中的“virtual history”相关量，
    // 常用于观察 VMB（Virtual Motion Based reprojection）是否在合理工作。
    // 对镜面/半镜面反射来说，这是非常关键的内部指标。
    // =====================================================================================
    else if( viewportIndex == 7 && gHasSpecular )
    {
        Text::Print_ch( 'V', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'T', textState );
        Text::Print_ch( 'U', textState );
        Text::Print_ch( 'A', textState );
        Text::Print_ch( 'L', textState );
        Text::Print_ch( ' ', textState );
        Text::Print_ch( 'H', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'S', textState );
        Text::Print_ch( 'T', textState );
        Text::Print_ch( 'O', textState );
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'Y', textState );

        result.xyz = data2.x * float( !isInf );
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 8: Diffuse accumulated frame count
    //
    // `data1.x` 表示 diffuse 历史长度。这里不是直接画“帧数越大越亮”，
    // 而是先做 1 - saturate(frames / maxFrames)，所以：
    //   - 帧数小 / 历史短 → 更偏热色，更显眼
    //   - 帧数大 / 历史稳定 → 更偏冷色
    //
    // 最顶部一条细带还会画一个 0~1 渐变标尺，便于阅读颜色映射。
    // =====================================================================================
    else if( viewportIndex == 8 && gHasDiffuse )
    {
        Text::Print_ch( 'D', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( ' ', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'A', textState );
        Text::Print_ch( 'M', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'S', textState );

        float f = 1.0 - saturate( data1.x / max( gMaxAccumulatedFrameNum, 1.0 ) );
        f = checkerboard && data1.x < 1.0 ? 0.75 : f;

        result.xyz = Color::ColorizeZucconi( viewportUv.y > 0.95 ? 1.0 - viewportUv.x : f * float( !isInf ) );
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 11: Specular accumulated frame count
    //
    // 与 diffuse frames 同理，只是读取的是 `data1.y`。
    // 这格通常比 diffuse 更敏感，因为 specular history 更容易受到 VMB、roughness、hitT 的影响。
    // =====================================================================================
    else if( viewportIndex == 11 && gHasSpecular )
    {
        Text::Print_ch( 'S', textState );
        Text::Print_ch( 'P', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'C', textState );
        Text::Print_ch( ' ', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'A', textState );
        Text::Print_ch( 'M', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'S', textState );

        float f = 1.0 - saturate( data1.y / max( gMaxAccumulatedFrameNum, 1.0 ) );
        f = checkerboard && data1.y < 1.0 ? 0.75 : f;

        result.xyz = Color::ColorizeZucconi( viewportUv.y > 0.95 ? 1.0 - viewportUv.x : f * float( !isInf ) );
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 12: Diffuse hit distance
    //
    // 可视化规则：
    //   - diff.w == 0        → 红色，表示没有有效 hit distance / miss / 无效输入
    //   - diff.w 不在 [0,1] → 品红色，表示归一化 hitT 异常
    //   - 其他               → 灰度显示 hitT
    //
    // 这是检查前端是否把 hit distance 正确归一化、是否出现 NaN/越界的第一现场。
    // =====================================================================================
    else if( viewportIndex == 12 && gHasDiffuse )
    {
        Text::Print_ch( 'D', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( ' ', textState );
        Text::Print_ch( 'H', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'T', textState );
        Text::Print_ch( 'T', textState );

        if( diff.w == 0.0 )
            result.xyz = float3( 1, 0, 0 );
        else
            result.xyz = diff.w != saturate( diff.w ) ? float3( 1, 0, 1 ) : diff.w;

        result.xyz *= float( !isInf );
        result.w = 1.0;
    }
    // =====================================================================================
    // [注解] Viewport 15: Specular hit distance
    //
    // 和 diffuse hitT 完全同构，只是改为读取 `spec.w`。
    // 对 specular 来说，这一格尤其重要，因为很多 temporal / virtual reprojection 逻辑都依赖 hitT。
    // =====================================================================================
    else if( viewportIndex == 15 && gHasSpecular )
    {
        Text::Print_ch( 'S', textState );
        Text::Print_ch( 'P', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'C', textState );
        Text::Print_ch( ' ', textState );
        Text::Print_ch( 'H', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'T', textState );
        Text::Print_ch( 'T', textState );

        if( spec.w == 0.0 )
            result.xyz = float3( 1, 0, 0 );
        else
            result.xyz = spec.w != saturate( spec.w ) ? float3( 1, 0, 1 ) : spec.w;

        result.xyz *= float( !isInf );
        result.w = 1.0;
    }

    // =====================================================================================
    // [注解] 文本前景处理
    //
    // 文字是后叠加在结果图上的。为了保证可读性：
    //   - 根据当前底色亮度自动决定是否把颜色往反色方向推
    //   - hitT 两个窗口里的文字固定用中灰，避免红/品红背景上对比过强
    // =====================================================================================
    if( Text::IsForeground( textState ) )
    {
        float lum = Color::Luminance( result.xyz );
        result.xyz = lerp( 0.0, 1.0 - result.xyz, saturate( abs( lum - 0.5 ) / 0.25 ) ) ;

        if( viewportIndex == 12 || viewportIndex == 15 )
            result.xyz = 0.5;
    }

    // [注解] 输出最终调试颜色
    gOut_Validation[ pixelPos ] = result;
}
