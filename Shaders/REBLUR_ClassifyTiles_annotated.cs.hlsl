/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// =====================================================================================
// [注解] 此 shader 的核心功能：REBLUR ClassifyTiles（Tile 分类）
//
// 这是 REBLUR 管线最前面的轻量级预处理 pass 之一。
// 它不做真正的降噪，而是先按 16x16 像素块（tile）统计：
//
//   - 这一整块是否都已经超出 `gDenoisingRange`
//   - 如果整块都超出范围，就把它标成 sky / empty tile
//   - 后续的 PrePass / TemporalAccumulation / Blur / PostBlur 等 pass
//     就可以通过 `gIn_Tiles` 直接跳过这块区域，减少无意义计算
//
// 也就是说，这个 pass 的作用本质上是：
//   **用极低成本提前找出“完全不需要降噪”的 tile**
//
// 这里的 `isSky` 更准确地说并不一定字面等于“天空材质”，
// 而是表示：该 tile 中所有像素的 `viewZ` 都大于 `gDenoisingRange`，
// 因而对 REBLUR 来说可以视为背景 / 无限远区域。
// =====================================================================================

// [注解] NRD 基础定义：导出宏、资源访问宏、基础类型
#include "NRD.hlsli"

// [注解] 数学库。这个文件里直接用到的不多，但 NRD shader 往往仍会统一包含它。
#include "ml.hlsli"


// [注解] REBLUR 全局配置：
//        这里会给出 `REBLUR_TILE_TYPE`、共享常量结构等基础定义。
#include "REBLUR_Config.hlsli"

// [注解] 当前 pass 的资源绑定定义：
//        输入：`gIn_ViewZ`
//        输出：`gOut_Tiles`
//        线程组逻辑上的 tile 大小为 16x16
#include "REBLUR_ClassifyTiles.resources.hlsli"

// [注解] 通用 helper，主要提供 `WithRectOrigin`、`UnpackViewZ` 等工具。
#include "Common.hlsli"

// =====================================================================================
// [注解] shared memory：用于汇总当前线程组对整个 tile 的统计结果
//
// 每个线程会统计自己负责的 8 个像素里，有多少像素超出 `gDenoisingRange`；
// 然后通过原子加把结果累加到 `s_Sum`。
// 最终如果 `s_Sum == 256`，说明整个 16x16 tile 的 256 个像素都可跳过。
// =====================================================================================
groupshared int s_Sum;

// =====================================================================================
// [注解] 线程组尺寸是 8x4，而不是 16x16
//
// 看起来有点反直觉，但这是因为：
//   - 一个线程不只处理 1 个像素
//   - 而是处理一个 2x4 的小块，共 8 个像素
//
// 所以：
//   - X 方向：8 个线程 * 每线程 2 像素 = 16 像素
//   - Y 方向：4 个线程 * 每线程 4 像素 = 16 像素
//
// 这样一个线程组正好覆盖一个完整的 16x16 tile，共 256 个像素。
// =====================================================================================
[numthreads( 8, 4, 1 )]
NRD_EXPORT void NRD_CS_MAIN( uint2 threadPos : SV_GroupThreadId, uint2 tilePos : SV_GroupId, uint threadIndex : SV_GroupIndex )
{
    // [注解] 由组内第一个线程清零 shared 累加器
    if( threadIndex == 0 )
        s_Sum = 0;

    // [注解] 确保所有线程都看到已经清零后的 `s_Sum`
    GroupMemoryBarrierWithGroupSync();

    // =====================================================================================
    // [注解] 为当前线程分配它负责的 2x4 像素子块左上角
    //
    // `tilePos * 16`：定位到当前 16x16 tile 的左上角
    // `threadPos * uint2( 2, 4 )`：把组内线程映射到 tile 内的 2x4 子块
    // =====================================================================================
    uint2 pixelPos = tilePos * 16 + threadPos * uint2( 2, 4 );
    int sum = 0;

    // [注解] 两层小循环遍历该线程负责的 8 个像素
    [unroll]
    for( uint i = 0; i < 2; i++ )
    {
        [unroll]
        for( uint j = 0; j < 4; j++ )
        {
            uint2 pos = pixelPos + uint2( i, j );

            // [注解] 读取并解包深度。
            //        `WithRectOrigin` 用于把局部矩形坐标转换到实际纹理坐标。
            float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pos ) ] );

            // [注解] 若当前像素深度超过降噪范围，则把它视为“可跳过像素”。
            //        这里只统计数量，不记录更复杂的类别。
            sum += viewZ > gDenoisingRange ? 1 : 0;
        }
    }

    // [注解] 把当前线程统计的结果原子累加到组共享总和里
    InterlockedAdd( s_Sum, sum );

    // [注解] 等待所有线程都完成统计
    GroupMemoryBarrierWithGroupSync();

    // =====================================================================================
    // [注解] 由组内第一个线程输出这个 tile 的最终分类结果
    //
    // `s_Sum == 256`：说明整块 16x16 的所有像素都超出降噪范围
    //   -> 输出 1.0，表示这是 sky / empty tile
    // 否则：
    //   -> 输出 0.0，表示后续 pass 仍需正常处理这块区域
    // =====================================================================================
    if( threadIndex == 0 )
    {
        float isSky = s_Sum == 256 ? 1.0 : 0.0;

        gOut_Tiles[ tilePos ] = isSky;
    }
}
