#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

/**
 * @brief CUDA에서 grid-stride 루프 형태로 병렬 처리를 단순화하기 위한 매크로.
 *        i가 0부터 n 전까지 (blockDim.x * gridDim.x) 간격으로 반복.
 *
 * @param i [in/out] 반복문에서 사용할 인덱스 변수(스레드 고유값)
 * @param n [in]     전체 반복 횟수
 */
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

/****************************************************************************
* 함수명  : ms_deform_attn_im2col_bilinear
* 설명    : 주어진 4차원 텐서(bottom_data)에서 (h, w) 부동소수점 위치에 대해
*           bilinear 보간을 수행하여 값을 샘플링한다. 
*
*           Bilinear 보간(Bilinear Interpolation)은 이미지나 2차원 신호에서,
*           정규화된 격자(grid) 좌표에 없는 임의의 점에서의 값을 추정하기 위해 
*           사용하는 보간 방법 중 하나입니다. 
*           
*           간단히 말해, 어떤 점 (x,y)가 네 개의 픽셀(또는 격자점) 사이에 위치할 때,
*           해당 점의 값을 주변 네 픽셀의 값과 거리(가중치)를 바탕으로 "선형(linear)"
*           형태로 섞어서(합쳐서) 추정하는 방식입니다.
*
* 매개변수:
*   bottom_data [in] - 입력 데이터(4차원 텐서를 1차원으로 펼친 배열)
*   height      [in] - 텐서의 높이
*   width       [in] - 텐서의 너비
*   nheads      [in] - 어텐션 헤드(head)의 수
*   channels    [in] - 채널 수
*   h           [in] - 보간할 세로 좌표(부동소수점)
*   w           [in] - 보간할 가로 좌표(부동소수점)
*   m           [in] - 현재 사용하는 어텐션 헤드의 인덱스
*   c           [in] - 현재 채널 인덱스
*
* 반환값  :
*   - bilinear 보간을 통해 계산된 샘플 값.
*
* 사용 예 :
*   - multi-scale deformable attention, deformable DETR, Vision Transformer 등에서
*     여러 스케일이나 위치에 대해 동적으로 특징 맵을 샘플링할 때 활용 가능.
*
****************************************************************************/
template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t* bottom_data,  // 4차원 텐서를 1차원으로 펼친 입력 데이터
    const int height,             // 텐서의 높이
    const int width,              // 텐서의 너비
    const int nheads,             // 어텐션 헤드의 개수
    const int channels,           // 채널 수
    scalar_t h,                   // 보간할 y좌표(세로, 부동소수점)
    scalar_t w,                   // 보간할 x좌표(가로, 부동소수점)
    const int m,                  // 어텐션 헤드 인덱스
    const int c                   // 채널 인덱스
)
{
    //----------------------------------------------------------------------------------
    // 1) (h, w)를 기준으로 가장 가까운 정수 좌표(아래쪽: h_low, w_low)와
    //    그 바로 윗좌표(위쪽: h_high, w_high)를 구한다.
    //----------------------------------------------------------------------------------
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    //----------------------------------------------------------------------------------
    // 2) (h, w)의 소수 부분(lh, lw)과 그 보완(hh, hw)을 구한다.
    //    예) lh = h - floor(h), hh = 1 - lh
    //----------------------------------------------------------------------------------
    scalar_t lh = h - h_low;  // 세로 좌표의 소수 부분
    scalar_t lw = w - w_low;  // 가로 좌표의 소수 부분
    scalar_t hh = 1 - lh;     // 세로 소수 부분의 보완
    scalar_t hw = 1 - lw;     // 가로 소수 부분의 보완

    //----------------------------------------------------------------------------------
    // 3) bilinear 보간은 네 개의 인접 픽셀(좌상단, 우상단, 좌하단, 우하단) 값을 활용한다.
    //    각각 v1, v2, v3, v4로 표기한다. 텐서 범위를 벗어나는 경우 0으로 처리한다.
    //----------------------------------------------------------------------------------
    scalar_t v1 = 0;  // 좌상단 픽셀 값
    if (h_low >= 0 && w_low >= 0)
    {
        int ptr1 = h_low * width * nheads * channels
                 + w_low * nheads * channels
                 + m * channels
                 + c;
        v1 = bottom_data[ptr1];
    }

    scalar_t v2 = 0;  // 우상단 픽셀 값
    if (h_low >= 0 && w_high <= width - 1)
    {
        int ptr2 = h_low * width * nheads * channels
                 + w_high * nheads * channels
                 + m * channels
                 + c;
        v2 = bottom_data[ptr2];
    }

    scalar_t v3 = 0;  // 좌하단 픽셀 값
    if (h_high <= height - 1 && w_low >= 0)
    {
        int ptr3 = h_high * width * nheads * channels
                 + w_low * nheads * channels
                 + m * channels
                 + c;
        v3 = bottom_data[ptr3];
    }

    scalar_t v4 = 0;  // 우하단 픽셀 값
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        int ptr4 = h_high * width * nheads * channels
                 + w_high * nheads * channels
                 + m * channels
                 + c;
        v4 = bottom_data[ptr4];
    }

    //----------------------------------------------------------------------------------
    // 4) bilinear 보간 가중치 계산
    //    - w1: 좌상단에 대한 가중치
    //    - w2: 우상단에 대한 가중치
    //    - w3: 좌하단에 대한 가중치
    //    - w4: 우하단에 대한 가중치
    //
    //    w1 = hh * hw
    //    w2 = hh * lw
    //    w3 = lh * hw
    //    w4 = lh * lw
    //----------------------------------------------------------------------------------
    scalar_t w1 = hh * hw;
    scalar_t w2 = hh * lw;
    scalar_t w3 = lh * hw;
    scalar_t w4 = lh * lw;

    //----------------------------------------------------------------------------------
    // 5) 최종 보간 값 = (v1*w1 + v2*w2 + v3*w3 + v4*w4)
    //----------------------------------------------------------------------------------
    scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    //----------------------------------------------------------------------------------
    // 6) 계산된 결과 반환
    //----------------------------------------------------------------------------------
    return val;
}


/**
 * @brief Deformable Attention의 역전파 단계에서 그라디언트 가중치를 계산하는 함수.
 *        주어진 좌표(h, w)와 그리드 위치(gh, gw)에 따라 bilinear 보간을 기반으로
 *        가중치를 계산하여 반환한다.
 *
 * @tparam scalar_t    실수(부동소수점) 타입 템플릿 파라미터 (예: float, double 등)
 *
 * @param h            [in]  샘플링된 y좌표 (부동소수점)
 * @param w            [in]  샘플링된 x좌표 (부동소수점)
 * @param gh           [in]  그리드의 y좌표 인덱스 (정수)
 * @param gw           [in]  그리드의 x좌표 인덱스 (정수)
 * @param height       [in]  피처 맵의 높이
 * @param width        [in]  피처 맵의 너비
 *
 * @return             bilinear 보간을 통한 그라디언트 가중치
 *
 * @details
 *  - 입력 좌표(h, w)가 피처 맵의 범위를 벗어나는 경우, 가중치는 0으로 설정된다.
 *  - 좌표(h, w)의 주변 4개의 픽셀(좌상단, 우상단, 좌하단, 우하단)에 대한 가중치를 계산한다.
 *  - 그리드 위치(gh, gw)가 해당 픽셀 중 하나에 해당하면, 해당 가중치를 반환한다.
 *  - Deformable Attention의 역전파에서 각 위치의 기여도를 계산하는 데 사용된다.
 */
template <typename scalar_t>
__device__ scalar_t ms_deform_attn_get_gradient_weight(
    scalar_t h, 
    scalar_t w,
    const int gh, 
    const int gw,
    const int height,
    const int width
) {
    //--------------------------------------------------------------------------
    // 입력 좌표(h, w)가 피처 맵의 유효 범위를 벗어나는 경우, 가중치는 0
    //--------------------------------------------------------------------------
    if (h <= -1 || h >= height || w <= -1 || w >= width)
    {
        return static_cast<scalar_t>(0);
    }

    //--------------------------------------------------------------------------
    // 좌표(h, w)를 기준으로 주변 정수 좌표(h_low, w_low)와 그 상위 좌표(h_high, w_high)를 계산
    //--------------------------------------------------------------------------
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    //--------------------------------------------------------------------------
    // 초기 가중치(weight)를 0으로 설정
    //--------------------------------------------------------------------------
    scalar_t weight = 0;

    //--------------------------------------------------------------------------
    // 각 그리드 위치(gh, gw)에 따라 가중치를 계산
    //--------------------------------------------------------------------------
    if (gh == h_low && gw == w_low)
        weight = (gh + 1 - h) * (gw + 1 - w);
    if (gh == h_low && gw == w_high)
        weight = (gh + 1 - h) * (w + 1 - gw);
    if (gh == h_high && gw == w_low)
        weight = (h + 1 - gh) * (gw + 1 - w);
    if (gh == h_high && gw == w_high)
        weight = (h + 1 - gh) * (w + 1 - gw);

    //--------------------------------------------------------------------------
    // 계산된 가중치 반환
    //--------------------------------------------------------------------------
    return weight;
}

/**
 * @brief Deformable Attention의 역전파 단계에서 특정 좌표에 대한 그라디언트 가중치를 계산하는 함수.
 *        주어진 좌표(h, w)와 그리드 위치(gh, gw)에 따라 bilinear 보간을 기반으로
 *        가중치를 계산하여 반환한다.
 *
 * @tparam scalar_t    실수(부동소수점) 타입 템플릿 파라미터 (예: float, double 등)
 *
 * @param h            [in]  샘플링된 y좌표 (부동소수점)
 * @param w            [in]  샘플링된 x좌표 (부동소수점)
 * @param m            [in]  현재 어텐션 헤드의 인덱스
 * @param c            [in]  현재 채널의 인덱스
 * @param height       [in]  피처 맵의 높이
 * @param width        [in]  피처 맵의 너비
 * @param nheads       [in]  어텐션 헤드의 수
 * @param channels     [in]  채널 수
 * @param bottom_data  [in]  입력 데이터 배열 (4차원 텐서를 1차원으로 펼친 형태)
 * @param bp_dir       [in]  그라디언트 방향 (1: w 방향, 0: h 방향)
 *
 * @return             bilinear 보간을 통한 그라디언트 가중치
 *
 * @details
 *  - 입력 좌표(h, w)가 피처 맵의 범위를 벗어나는 경우, 가중치는 0으로 설정된다.
 *  - 좌표(h, w)의 주변 4개의 픽셀(좌상단, 우상단, 좌하단, 우하단)에 대한 값을 가져온다.
 *  - bp_dir 값에 따라 w 방향 또는 h 방향에 대한 그라디언트 가중치를 계산한다.
 *  - Deformable Attention의 역전파에서 각 위치의 기여도를 정확하게 반영하기 위해 사용된다.
 *
 * @note
 *  - 이 함수는 bilinear 보간의 역전파 단계에서 사용되며, 
 *    각 픽셀의 기여도를 계산하는 데 중요한 역할을 한다.
 *  - 템플릿 파라미터 `scalar_t`를 통해 다양한 부동소수점 타입에서 사용 가능하다.
 */
template <typename scalar_t>
__device__ scalar_t ms_deform_attn_get_coordinate_weight(
    scalar_t h, 
    scalar_t w, 
    const int m, 
    const int c,
    const int height, 
    const int width, 
    const int nheads, 
    const int channels,
    const scalar_t *bottom_data, 
    const int bp_dir)
{
    //--------------------------------------------------------------------------
    // 입력 좌표(h, w)가 피처 맵의 유효 범위를 벗어나는 경우, 가중치는 0
    //--------------------------------------------------------------------------
    if (h <= -1 || h >= height || w <= -1 || w >= width)
    {
      // empty
      return 0;
    }

    //--------------------------------------------------------------------------
    // 좌표(h, w)를 기준으로 주변 정수 좌표(h_low, w_low)와 그 상위 좌표(h_high, w_high)를 계산
    //--------------------------------------------------------------------------
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    //--------------------------------------------------------------------------
    // 초기 가중치(weight)를 0으로 설정
    //--------------------------------------------------------------------------
    scalar_t weight = 0;

    //--------------------------------------------------------------------------
    // 주변 4개 픽셀 값(v1, v2, v3, v4)을 가져온다.
    //   v1: 좌상단 (h_low, w_low)
    //   v2: 우상단 (h_low, w_high)
    //   v3: 좌하단 (h_high, w_low)
    //   v4: 우하단 (h_high, w_high)
    //--------------------------------------------------------------------------
    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        int ptr1 = h_low * width * nheads * channels 
                 + w_low * nheads * channels 
                 + m * channels 
                 + c;
        v1 = bottom_data[ptr1];
    }

    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        int ptr2 = h_low * width * nheads * channels 
                 + w_high * nheads * channels 
                 + m * channels 
                 + c;
        v2 = bottom_data[ptr2];
    }

    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        int ptr3 = h_high * width * nheads * channels 
                 + w_low * nheads * channels 
                 + m * channels 
                 + c;
        v3 = bottom_data[ptr3];
    }

    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        int ptr4 = h_high * width * nheads * channels 
                 + w_high * nheads * channels 
                 + m * channels 
                 + c;
        v4 = bottom_data[ptr4];
    }

    //--------------------------------------------------------------------------
    // bp_dir에 따라 w 방향 또는 h 방향의 그라디언트 가중치를 계산
    //--------------------------------------------------------------------------
    if (bp_dir == 1)
    {
        // w 방향 그라디언트
        if (h_low >= 0 && w_low >= 0)
            weight += -1 * (w_low + 1 - w) * v1;
        if (h_low >= 0 && w_high <= width - 1)
            weight += -1 * (w - w_low) * v2;
        if (h_high <= height - 1 && w_low >= 0)
            weight += (w_low + 1 - w) * v3;
        if (h_high <= height - 1 && w_high <= width - 1)
            weight += (w - w_low) * v4;
    }
    else if (bp_dir == 0)
    {
        // h 방향 그라디언트
        if (h_low >= 0 && w_low >= 0)
            weight += -1 * (h_low + 1 - h) * v1;
        if (h_low >= 0 && w_high <= width - 1)
            weight += (h_low + 1 - h) * v2;
        if (h_high <= height - 1 && w_low >= 0)
            weight += -1 * (h - h_low) * v3;
        if (h_high <= height - 1 && w_high <= width - 1)
            weight += (h - h_low) * v4;
    }

    //--------------------------------------------------------------------------
    // 계산된 가중치 반환
    //--------------------------------------------------------------------------
    return weight;
}

/**
 * @brief 여러 레벨(feature levels)에 걸쳐 정의된 특징 맵(data_value)에서
 *        주어진 샘플링 좌표(data_sampling_loc)와 어텐션 가중치(data_attn_weight)를 통해
 *        bilinear 보간 값을 계산하여 data_col에 저장하는 CUDA 커널.
 *
 * @tparam scalar_t    실수(부동소수점) 타입 템플릿 파라미터 (예: float, double 등)
 *
 * @param n                       [in]  전체 스레드(출력 픽셀) 개수
 * @param data_value              [in]  입력 특징 맵 
 *                                     (shape: [batch_size, spatial_size, num_heads, channels])
 * @param data_spatial_shapes     [in]  각 레벨의 (height, width) 정보
 *                                     (shape: [num_levels, 2])
 * @param data_level_start_index  [in]  각 레벨 시작의 spatial 인덱스 (shape: [num_levels])
 * @param data_sampling_loc       [in]  샘플링 위치 정보 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                               num_levels, num_point, 2])
 * @param data_attn_weight        [in]  어텐션 가중치 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                               num_levels, num_point])
 * @param batch_size              [in]  배치 크기
 * @param spatial_size            [in]  전체 공간 픽셀 수(모든 레벨 합)
 * @param num_heads               [in]  어텐션 헤드의 수
 * @param channels                [in]  채널 수
 * @param num_levels              [in]  피처 레벨의 수
 * @param num_query               [in]  쿼리(토큰)의 수
 * @param num_point               [in]  각 레벨별 샘플링할 포인트(위치) 수
 * @param data_col                [out] 계산된 결과를 저장할 배열
 *                                     (shape: [num_levels * num_point, 
 *                                              batch_size, num_query, 
 *                                              num_heads, channels])
 *
 * @note  커널 함수이므로 반환값은 없다.
 *
 * @details
 *    다중 스케일 피처맵을 활용하여 역동적인 위치 샘플링 및 가중 합 수행 가능.
 */
template <typename scalar_t> __global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *data_col)
{
    //---------------------------------------------------------------------------
    // grid-stride 루프: 각 스레드별로 index(출력 픽셀) 범위를 나눠 담당.
    // 
    // launch batch_size * num_levels * num_query * num_point * channels cores
    // data_value: batch_size, spatial_size, num_heads, channels
    // data_sampling_loc: batch_size, num_query, num_heads, num_levels, num_point, 2
    // data_attn_weight: batch_size, num_query, num_heads, num_levels, num_point
    // data_col: num_levels*num_point, batch_size, num_query, num_heads, channels
    //---------------------------------------------------------------------------
    CUDA_KERNEL_LOOP(index, n)
    {
        //-----------------------------------------------------------------------
        // index(출력 픽셀)로부터 (b_col, l_col, q_col, p_col, c_col) 추출
        //   b_col : 배치 인덱스
        //   l_col : 레벨 인덱스
        //   q_col : 쿼리 인덱스
        //   p_col : 포인트 인덱스
        //   c_col : 채널 인덱스
        //-----------------------------------------------------------------------
        const int c_col = index % channels;
        const int p_col = (index / channels) % num_point;
        const int q_col = (index / channels / num_point) % num_query;
        const int l_col = (index / channels / num_point / num_query) % num_levels;
        const int b_col =  index / channels / num_point / num_query / num_levels;

        //-----------------------------------------------------------------------
        // 현재 레벨(l_col)에서의 시작 인덱스(level_start_id)와
        // 해당 레벨 피처맵의 높이(spatial_h), 너비(spatial_w)
        //-----------------------------------------------------------------------
        const int level_start_id = data_level_start_index[l_col];
        const int spatial_h      = data_spatial_shapes[l_col * 2];
        const int spatial_w      = data_spatial_shapes[l_col * 2 + 1];

        //-----------------------------------------------------------------------
        // 출력 배열(data_col)에 대한 포인터 계산
        // data_col:
        //   [l_col, p_col, b_col, q_col, (head: i), c_col]
        //-----------------------------------------------------------------------
        scalar_t *data_col_ptr = data_col
            + ( c_col
              + channels * 0  // head = 0부터 시작, 아래 for문에서 head마다 += channels
              + channels * num_heads * q_col
              + channels * num_heads * num_query * b_col
              + channels * num_heads * num_query * batch_size * p_col
              + channels * num_heads * num_query * batch_size * num_point * l_col );

        //-----------------------------------------------------------------------
        // 입력 피처맵(data_value)의 접근 포인터
        //   data_value: [batch_size, spatial_size, num_heads, channels]
        //-----------------------------------------------------------------------
        const scalar_t *data_value_ptr = data_value
            + ( b_col * spatial_size * num_heads * channels
              + level_start_id * num_heads * channels );

        //-----------------------------------------------------------------------
        // data_sampling_loc, data_attn_weight 포인터
        //   (batch, query)에 따라 offset을 맞춰줌
        //-----------------------------------------------------------------------
        const scalar_t *data_sampling_loc_ptr = data_sampling_loc
            + ( b_col * num_query * num_heads * num_levels * num_point * 2
              + q_col * num_heads * num_levels * num_point * 2 );

        const scalar_t *data_attn_weight_ptr = data_attn_weight
            + ( b_col * num_query * num_heads * num_levels * num_point
              + q_col * num_heads * num_levels * num_point );

        //-----------------------------------------------------------------------
        // i: head 인덱스 (0 ~ num_heads-1)
        //-----------------------------------------------------------------------
        for (int i = 0; i < num_heads; ++i)
        {
            //-------------------------------------------------------------------
            // 샘플링 위치 loc_h, loc_w 인덱스
            //   - data_sampling_loc은 (..., 2) 구조이므로
            //     w, h 순서 확인이 필요 (현재 코드는 w -> data_loc_w_ptr,
            //     h -> data_loc_h_ptr 형태)
            //-------------------------------------------------------------------
            const int data_loc_h_ptr = i * num_levels * num_point * 2
                                     + l_col * num_point * 2
                                     + p_col * 2
                                     + 1;
            const int data_loc_w_ptr = i * num_levels * num_point * 2
                                     + l_col * num_point * 2
                                     + p_col * 2
                                     + 0;

            // 어텐션 가중치 인덱스
            const int data_weight_ptr = i * num_levels * num_point
                                      + l_col * num_point
                                      + p_col;

            // loc_h, loc_w: (0~1) 정규화 좌표 -> 실제 픽셀 좌표로 변환
            const scalar_t loc_h = data_sampling_loc_ptr[data_loc_h_ptr];
            const scalar_t loc_w = data_sampling_loc_ptr[data_loc_w_ptr];

            // 어텐션 가중치
            const scalar_t weight = data_attn_weight_ptr[data_weight_ptr];

            // 보간 결과값을 임시 저장할 변수
            scalar_t val = static_cast<scalar_t>(0);

            //-------------------------------------------------------------------
            // 실제 이미지 좌표 = (정규화 좌표 * 크기) - 0.5
            //   - 0.5는 grid 샘플링 기준점 보정을 위한 것
            //-------------------------------------------------------------------
            const scalar_t h_im = loc_h * spatial_h - static_cast<scalar_t>(0.5);
            const scalar_t w_im = loc_w * spatial_w - static_cast<scalar_t>(0.5);

            //-------------------------------------------------------------------
            // bilinear 보간을 적용할 범위 검사
            //-------------------------------------------------------------------
            if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
            {
                val = ms_deform_attn_im2col_bilinear(
                    data_value_ptr,
                    spatial_h,
                    spatial_w,
                    num_heads,
                    channels,
                    h_im,
                    w_im,
                    i,         // head index
                    c_col      // channel index
                );
            }

            //-------------------------------------------------------------------
            // 최종 값 = 보간 값(val) × 어텐션 가중치(weight)
            // data_col_ptr는 head별로 channels만큼 건너뜀
            //-------------------------------------------------------------------
            *data_col_ptr = val * weight;
            data_col_ptr += channels;
        }
    }
}
/******************************************************************************
 * @brief 여러 레벨(feature levels)에 걸쳐 정의된 특징 맵(data_value)에서
 *        샘플링된 위치(data_sampling_loc)와 어텐션 가중치(data_attn_weight)를
 *        사용하여 그라디언트 값을 계산하고, 이를 grad_value에 누적하는 CUDA 커널.
 *        주로 Deformable DETR, Multi-Scale Deformable Attention 등에서 사용됨.
 *
 * @tparam scalar_t    실수(부동소수점) 타입 템플릿 파라미터 (예: float, double 등)
 *
 * @param n                       [in]  전체 스레드(출력 픽셀) 개수
 * @param data_col                [in]  입력 데이터 컬럼 
 *                                     (shape: [batch_size, num_query, num_heads, channels])
 * @param data_spatial_shapes     [in]  각 레벨의 (height, width) 정보
 *                                     (shape: [num_levels, 2])
 * @param data_level_start_index  [in]  각 레벨 시작의 spatial 인덱스 (shape: [num_levels])
 * @param data_sampling_loc       [in]  샘플링 위치 정보 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point, 2])
 * @param data_attn_weight        [in]  어텐션 가중치 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point])
 * @param batch_size              [in]  배치 크기
 * @param spatial_size            [in]  전체 공간 픽셀 수(모든 레벨 합)
 * @param num_heads               [in]  어텐션 헤드의 수
 * @param channels                [in]  채널 수
 * @param num_levels              [in]  피처 레벨의 수
 * @param num_query               [in]  쿼리(토큰)의 수
 * @param num_point               [in]  각 레벨별 샘플링할 포인트(위치) 수
 * @param grad_value              [out] 계산된 그라디언트 값을 저장할 배열
 *                                     (shape: [batch_size, spatial_size, num_heads, channels])
 *
 * @return                      없음 (커널 함수이므로 반환값 없음)
 *
 * @details
 *  - 이 커널은 forward pass에서 계산된 data_col과 샘플링 위치, 어텐션 가중치를
 *    사용하여 backward pass에서의 그라디언트를 계산하고, 이를 grad_value에 누적한다.
 *  - 각 스레드는 고유한 인덱스를 기반으로 data_col의 특정 위치를 처리하며,
 *    해당 위치의 그라디언트를 주변의 grad_value 위치에 bilinear 보간 가중치와 함께
 *    누적한다.
 *  - bilinear 보간을 사용하여 그라디언트가 주변 픽셀에 올바르게 분배되도록 한다.
 *
 * @note
 *  - 이 커널은 Deformable Attention의 역전파 단계에서 사용되며, 
 *    각 샘플링 위치에 대한 그라디언트를 정확하게 계산하고 누적하는 데 필수적이다.
 *  - atomicAdd를 사용하여 여러 스레드가 동일한 grad_value 위치에 접근할 때의 동시성을 처리한다.
 ******************************************************************************/
template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel(
    const int n,
    const scalar_t *data_col,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *grad_value)
{
    //---------------------------------------------------------------------------
    // grid-stride 루프: 각 스레드별로 index(출력 픽셀) 범위를 나눠 담당.
    //---------------------------------------------------------------------------
    CUDA_KERNEL_LOOP(index, n)
    {
        //-----------------------------------------------------------------------
        // index(출력 픽셀)로부터 (b_col, l_col, q_col, p_col, m_col, c_col) 추출
        //   b_col : 배치 인덱스
        //   l_col : 레벨 인덱스
        //   q_col : 쿼리 인덱스
        //   p_col : 포인트 인덱스
        //   m_col : 헤드 인덱스
        //   c_col : 채널 인덱스
        //-----------------------------------------------------------------------
        const int c_col = index % channels;
        const int m_col = (index / channels) % num_heads;
        const int p_col = (index / channels / num_heads) % num_point;
        const int q_col = (index / channels / num_heads / num_point) % num_query;
        const int l_col = (index / channels / num_heads / num_point / num_query) % num_levels;
        const int b_col = index / channels / num_heads / num_point / num_query / num_levels;

        //-----------------------------------------------------------------------
        // 현재 레벨(l_col)에서의 시작 인덱스(level_start_id)와
        // 해당 레벨 피처맵의 높이(spatial_h), 너비(spatial_w) 구하기
        //-----------------------------------------------------------------------
        const int level_start_id = data_level_start_index[l_col];
        const int spatial_h = data_spatial_shapes[l_col * 2];
        const int spatial_w = data_spatial_shapes[l_col * 2 + 1];

        //-----------------------------------------------------------------------
        // data_col에서 현재 위치(c_col, m_col, q_col, b_col)를 기반으로 값 가져오기
        // data_col 구조: [batch_size, num_query, num_heads, channels]
        //-----------------------------------------------------------------------
        const scalar_t col = data_col[
            c_col
            + channels * m_col
            + channels * num_heads * q_col
            + channels * num_heads * num_query * b_col
        ];

        //-----------------------------------------------------------------------
        // 샘플링 위치 및 어텐션 가중치 인덱스 계산
        // data_sampling_loc: [batch_size, num_query, num_heads, num_levels, num_point, 2]
        // data_attn_weight: [batch_size, num_query, num_heads, num_levels, num_point]
        //-----------------------------------------------------------------------
        int sampling_ptr = b_col * num_query * num_heads * num_levels * num_point
                        + q_col * num_heads * num_levels * num_point
                        + m_col * num_levels * num_point
                        + l_col * num_point
                        + p_col;
        const scalar_t sampling_x = data_sampling_loc[2 * sampling_ptr] * spatial_w - 0.5;
        const scalar_t sampling_y = data_sampling_loc[2 * sampling_ptr + 1] * spatial_h - 0.5;
        const scalar_t attn_weight = data_attn_weight[sampling_ptr];

        //-----------------------------------------------------------------------
        // 현재 그라디언트 값 계산
        // col은 forward pass에서의 값이며, 이를 어텐션 가중치와 곱하여 역전파 그라디언트를 계산
        //-----------------------------------------------------------------------
        const scalar_t cur_top_grad = col * attn_weight;

        //-----------------------------------------------------------------------
        // 샘플링 위치의 정수 부분 추출
        // (정수 부분을 기반으로 주변 픽셀에 그라디언트를 분배)
        //-----------------------------------------------------------------------
        const int cur_h = (int)sampling_y;
        const int cur_w = (int)sampling_x;

        //-----------------------------------------------------------------------
        // 주변 픽셀에 대한 그라디언트 가중치 계산 및 누적
        // bilinear 보간의 역전파 단계
        //-----------------------------------------------------------------------
        for (int dy = -2; dy <= 2; dy++)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                //-------------------------------------------------------------------
                // (cur_h + dy, cur_w + dx)가 유효한 위치인지 확인
                // 그리고 bilinear 보간의 범위 내에 있는지 확인
                //-------------------------------------------------------------------
                if (cur_h + dy >= 0 && cur_h + dy < spatial_h &&
                    cur_w + dx >= 0 && cur_w + dx < spatial_w &&
                    abs(sampling_y - (cur_h + dy)) < 1 &&
                    abs(sampling_x - (cur_w + dx)) < 1)
                {
                    //-------------------------------------------------------------------
                    // grad_value에서의 현재 위치 계산
                    // grad_value 구조: [batch_size, spatial_size, num_heads, channels]
                    //-------------------------------------------------------------------
                    int cur_bottom_grad_pos = b_col * spatial_size * num_heads * channels
                                            + (level_start_id + (cur_h + dy) * spatial_w + (cur_w + dx)) * num_heads * channels
                                            + m_col * channels
                                            + c_col;

                    //-------------------------------------------------------------------
                    // 그라디언트 가중치 계산
                    // ms_deform_attn_get_gradient_weight 함수 사용
                    //-------------------------------------------------------------------
                    scalar_t weight = ms_deform_attn_get_gradient_weight(
                        sampling_y,
                        sampling_x,
                        cur_h + dy,
                        cur_w + dx,
                        spatial_h,
                        spatial_w
                    );

                    //-------------------------------------------------------------------
                    // grad_value에 그라디언트 가중치 * 현재 그라디언트 값을 atomicAdd로 누적
                    // 여러 스레드가 동일 위치에 접근할 수 있으므로 atomicAdd 사용
                    //-------------------------------------------------------------------
                    atomicAdd(grad_value + cur_bottom_grad_pos, weight * cur_top_grad);
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_coord_gpu_kernel(
    const int n,
    const scalar_t *data_col,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *grad_sampling_loc, 
    scalar_t *grad_attn_weight)
{
    //--------------------------------------------------------------------------
    // grid-stride 루프: 각 스레드별로 index(출력 픽셀) 범위를 나눠 담당.
    //--------------------------------------------------------------------------
    CUDA_KERNEL_LOOP(index, n)
    {
        //--------------------------------------------------------------------------
        // 현재 위치에서 사용할 임시 변수 초기화
        //--------------------------------------------------------------------------
        scalar_t val = 0;
        scalar_t wval = 0;

        //--------------------------------------------------------------------------
        // index를 기반으로 (loc_c, k, l, m, q, b)를 역으로 계산
        //   loc_c : 좌표 방향 (0: x, 1: y)
        //   k     : 포인트 인덱스
        //   l     : 레벨 인덱스
        //   m     : 헤드 인덱스
        //   q     : 쿼리 인덱스
        //   b     : 배치 인덱스
        //--------------------------------------------------------------------------
        const int loc_c = index % 2;
        const int k = (index / 2) % num_point;
        const int l = (index / 2 / num_point) % num_levels;
        const int m = (index / 2 / num_point / num_levels) % num_heads;
        const int q = (index / 2 / num_point / num_levels / num_heads) % num_query;
        const int b = index / 2 / num_point / num_levels / num_heads / num_query;

        //--------------------------------------------------------------------------
        // 현재 레벨(l)의 시작 인덱스(level_start_id), 높이(spatial_h), 너비(spatial_w) 확인
        //--------------------------------------------------------------------------
        const int level_start_id = data_level_start_index[l];
        const int spatial_h = data_spatial_shapes[l * 2];
        const int spatial_w = data_spatial_shapes[l * 2 + 1];

        //--------------------------------------------------------------------------
        // data_col에서 현재 위치(m, q, b)에 해당하는 포인터 계산
        // data_col: [batch_size, num_query, num_heads, channels]
        //   -> (m, q, b)를 이용해 오프셋 계산
        //--------------------------------------------------------------------------
        const scalar_t *data_col_ptr = data_col
            + ( m * channels
              + q * channels * num_heads
              + b * channels * num_heads * num_query );

        //--------------------------------------------------------------------------
        // data_value에서 현재 레벨(level_start_id) 오프셋을 사용해 포인터 계산
        // data_value: [batch_size, spatial_size, num_heads, channels]
        //--------------------------------------------------------------------------
        const scalar_t *data_value_ptr = data_value
            + ( 0 * channels
              + level_start_id * channels * num_heads
              + b * channels * num_heads * spatial_size );

        scalar_t sampling_x = data_sampling_loc[(index / 2) * 2] * spatial_w - 0.5;
        scalar_t sampling_y = data_sampling_loc[(index / 2) * 2 + 1] * spatial_h - 0.5;
        const scalar_t attn_weight = data_attn_weight[index / 2];

        //--------------------------------------------------------------------------
        // 현재 포인트(k)에 대해 모든 채널(col_c)을 순회하며
        //   -> bilinear 보간 값(wval) 및 좌표 그라디언트(val)를 계산
        //--------------------------------------------------------------------------
        for (int col_c = 0; col_c < channels; col_c += 1)
        {
            // data_col 값
            const scalar_t col = data_col_ptr[col_c];

            // 범위 벗어나면 다시 샘플링 불가 상태로 세팅(-2)
            if (sampling_x <= -1 || sampling_y <= -1 || sampling_x >= spatial_w || sampling_y >= spatial_h)
            {
                sampling_x = sampling_y = -2;
            }
            else
            {
                // wval은 bilinear 보간으로 얻은 값들의 합(어텐션 가중치 고려 전)
                wval += col * ms_deform_attn_im2col_bilinear(
                            data_value_ptr,
                            spatial_h,
                            spatial_w,
                            num_heads,
                            channels,
                            sampling_y,
                            sampling_x,
                            m,
                            col_c
                        );
            }

            // val은 좌표 그라디언트(로컬 편미분)에 해당
            // ms_deform_attn_get_coordinate_weight를 통해 좌표에 대한 편미분 값(weight) 계산
            // 이후 col * attn_weight와 곱해 최종 기여도(val)에 더함
            const scalar_t weight = ms_deform_attn_get_coordinate_weight(
                sampling_y, sampling_x,
                m, col_c,
                spatial_h, spatial_w,
                num_heads, channels,
                data_value_ptr,
                loc_c
            );
            val += weight * col * attn_weight;
        }

        //--------------------------------------------------------------------------
        // x 방향(loc_c == 0)일 경우, spatial_w로 스케일링
        // y 방향(loc_c == 1)일 경우, spatial_h로 스케일링
        //--------------------------------------------------------------------------
        if (loc_c == 0)
        {
            val *= spatial_w;
        }
        else if (loc_c == 1)
        {
            val *= spatial_h;
        }

        //--------------------------------------------------------------------------
        // 계산된 좌표 그라디언트(val)를 grad_sampling_loc 배열에 저장
        //--------------------------------------------------------------------------
        grad_sampling_loc[index] = val;

        //--------------------------------------------------------------------------
        // x 방향(loc_c % 2 == 0)일 때 어텐션 가중치 그라디언트(grad_attn_weight) 갱신
        // (필요에 따라 y 방향 포함 등 구현 방식은 모델마다 다를 수 있음)
        //--------------------------------------------------------------------------
        if (loc_c % 2 == 0)
        {
            grad_attn_weight[index / 2] = wval;
        }
    }
}

/******************************************************************************
 * @brief Deformable Attention 메커니즘의 im2col 연산을 GPU에서 수행하는 CUDA 호스트 함수.
 *        입력 특징 맵과 샘플링 위치, 어텐션 가중치를 사용하여 data_col 배열을 계산합니다.
 *        주로 Deformable DETR, Multi-Scale Deformable Attention 등에서 사용됨.
 *
 * @tparam scalar_t    실수(부동소수점) 타입 템플릿 파라미터 (예: float, double 등)
 *
 * @param stream                  [in]  CUDA 스트림
 * @param data_value              [in]  입력 특징 맵 
 *                                     (shape: [batch_size, spatial_size, num_heads, channels])
 * @param data_spatial_shapes     [in]  각 레벨의 (height, width) 정보
 *                                     (shape: [num_levels, 2])
 * @param data_level_start_index  [in]  각 레벨 시작의 spatial 인덱스 (shape: [num_levels])
 * @param data_sampling_loc       [in]  샘플링 위치 정보 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point, 2])
 * @param data_attn_weight        [in]  어텐션 가중치 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point])
 * @param batch_size              [in]  배치 크기
 * @param spatial_size            [in]  전체 공간 픽셀 수(모든 레벨 합)
 * @param num_heads               [in]  어텐션 헤드의 수
 * @param channels                [in]  채널 수
 * @param num_levels              [in]  피처 레벨의 수
 * @param num_query               [in]  쿼리(토큰)의 수
 * @param num_point               [in]  각 레벨별 샘플링할 포인트(위치) 수
 * @param data_col                [out] 계산된 결과를 저장할 배열
 *                                     (shape: [batch_size, num_query, num_heads, channels])
 *
 * @return                      없음 (커널 함수이므로 반환값 없음)
 *
 * @details
 *  - 이 함수는 forward pass에서 사용되는 im2col 연산을 CUDA를 통해 GPU에서 수행합니다.
 *  - 각 스레드는 data_col의 특정 위치를 처리하며, bilinear 보간을 통해 데이터를 샘플링합니다.
 *  - `GET_BLOCKS(num_kernels)`와 `CUDA_NUM_THREADS`는 커널 실행 시 필요한 블록과 스레드 수를 계산합니다.
 *  - 커널 실행 후, CUDA 에러를 체크하여 문제 발생 시 에러 메시지를 출력합니다.
 *
 * @note
 *  - `GET_BLOCKS`와 `CUDA_NUM_THREADS`는 사전에 정의된 매크로로, GPU의 스레드 및 블록 설정을 조정합니다.
 *  - 이 함수는 Deformable DETR, Multi-Scale Deformable Attention 등에서 주로 사용됩니다.
 ******************************************************************************/
template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream,
                               const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size,
                               const int spatial_size,
                               const int num_heads,
                               const int channels,
                               const int num_levels,
                               const int num_query,
                               const int num_point,
                               scalar_t *data_col)
{
    //---------------------------------------------------------------------------
    // 커널 실행에 필요한 스레드 수 계산
    // num_kernels = batch_size * num_levels * num_query * num_point * channels
    //---------------------------------------------------------------------------
    const int num_kernels = batch_size * num_levels * num_query * num_point * channels;

    //---------------------------------------------------------------------------
    // ms_deformable_im2col_gpu_kernel 커널 실행
    //---------------------------------------------------------------------------
    ms_deformable_im2col_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
           0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight,
            batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);

    //---------------------------------------------------------------------------
    // CUDA 에러 체크
    //---------------------------------------------------------------------------
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}

/******************************************************************************
 * @brief Deformable Attention 메커니즘의 col2im 연산을 GPU에서 수행하는 CUDA 호스트 함수.
 *        입력 data_col과 샘플링 위치, 어텐션 가중치를 사용하여 grad_value 배열에 그라디언트를 누적합니다.
 *        주로 Deformable DETR, Multi-Scale Deformable Attention 등에서 사용됨.
 *
 * @tparam scalar_t    실수(부동소수점) 타입 템플릿 파라미터 (예: float, double 등)
 *
 * @param stream                  [in]  CUDA 스트림
 * @param data_col                [in]  입력 데이터 컬럼 
 *                                     (shape: [batch_size, num_query, num_heads, channels])
 * @param data_spatial_shapes     [in]  각 레벨의 (height, width) 정보
 *                                     (shape: [num_levels, 2])
 * @param data_level_start_index  [in]  각 레벨 시작의 spatial 인덱스 (shape: [num_levels])
 * @param data_sampling_loc       [in]  샘플링 위치 정보 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point, 2])
 * @param data_attn_weight        [in]  어텐션 가중치 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point])
 * @param batch_size              [in]  배치 크기
 * @param spatial_size            [in]  전체 공간 픽셀 수(모든 레벨 합)
 * @param num_heads               [in]  어텐션 헤드의 수
 * @param channels                [in]  채널 수
 * @param num_levels              [in]  피처 레벨의 수
 * @param num_query               [in]  쿼리(토큰)의 수
 * @param num_point               [in]  각 레벨별 샘플링할 포인트(위치) 수
 * @param grad_value              [out] 계산된 그라디언트 값을 저장할 배열
 *                                     (shape: [batch_size, spatial_size, num_heads, channels])
 *
 * @return                      없음 (커널 함수이므로 반환값 없음)
 *
 * @details
 *  - 이 함수는 backward pass에서 사용되는 col2im 연산을 CUDA를 통해 GPU에서 수행합니다.
 *  - 각 스레드는 grad_value의 특정 위치를 업데이트하며, bilinear 보간을 통해 그라디언트를 분배합니다.
 *  - `GET_BLOCKS(num_kernels)`와 `CUDA_NUM_THREADS`는 커널 실행 시 필요한 블록과 스레드 수를 계산합니다.
 *  - 커널 실행 후, CUDA 에러를 체크하여 문제 발생 시 에러 메시지를 출력합니다.
 *
 * @note
 *  - `GET_BLOCKS`와 `CUDA_NUM_THREADS`는 사전에 정의된 매크로로, GPU의 스레드 및 블록 설정을 조정합니다.
 *  - 이 함수는 Deformable DETR, Multi-Scale Deformable Attention 등에서 주로 사용됩니다.
 ******************************************************************************/
template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream,
                               const scalar_t *data_col,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size,
                               const int spatial_size,
                               const int num_heads,
                               const int channels,
                               const int num_levels,
                               const int num_query,
                               const int num_point,
                               scalar_t *grad_value)
{
    //---------------------------------------------------------------------------
    // 커널 실행에 필요한 스레드 수 계산
    // num_kernels = batch_size * num_levels * num_query * num_point * num_heads * channels
    //---------------------------------------------------------------------------
    const int num_kernels = batch_size * num_levels * num_query * num_point * num_heads * channels;

    //---------------------------------------------------------------------------
    // ms_deformable_col2im_gpu_kernel 커널 실행
    //---------------------------------------------------------------------------
    ms_deformable_col2im_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
           0, stream>>>(
            num_kernels,
            data_col,
            data_spatial_shapes,
            data_level_start_index,
            data_sampling_loc,
            data_attn_weight,
            batch_size,
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            grad_value);

    //---------------------------------------------------------------------------
    // CUDA 에러 체크
    //---------------------------------------------------------------------------
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
    }
}

/******************************************************************************
 * @brief Deformable Attention 메커니즘의 역전파 단계에서
 *        샘플링 위치와 어텐션 가중치에 대한 그라디언트를 계산하여
 *        grad_sampling_loc과 grad_attn_weight에 저장하는 CUDA 호스트 함수.
 *        주로 Deformable DETR, Multi-Scale Deformable Attention 등에서 사용됨.
 *
 * @tparam scalar_t    실수(부동소수점) 타입 템플릿 파라미터 (예: float, double 등)
 *
 * @param stream                  [in]  CUDA 스트림
 * @param data_col                [in]  입력 데이터 컬럼 
 *                                     (shape: [batch_size, num_query, num_heads, channels])
 * @param data_value              [in]  입력 특징 맵 
 *                                     (shape: [batch_size, spatial_size, num_heads, channels])
 * @param data_spatial_shapes     [in]  각 레벨의 (height, width) 정보
 *                                     (shape: [num_levels, 2])
 * @param data_level_start_index  [in]  각 레벨 시작의 spatial 인덱스 (shape: [num_levels])
 * @param data_sampling_loc       [in]  샘플링 위치 정보 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point, 2])
 * @param data_attn_weight        [in]  어텐션 가중치 
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                             num_levels, num_point])
 * @param batch_size              [in]  배치 크기
 * @param spatial_size            [in]  전체 공간 픽셀 수(모든 레벨 합)
 * @param num_heads               [in]  어텐션 헤드의 수
 * @param channels                [in]  채널 수
 * @param num_levels              [in]  피처 레벨의 수
 * @param num_query               [in]  쿼리(토큰)의 수
 * @param num_point               [in]  각 레벨별 샘플링할 포인트(위치) 수
 * @param grad_sampling_loc       [out] 계산된 그라디언트 샘플링 위치를 저장할 배열
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                              num_levels, num_point, 2])
 * @param grad_attn_weight        [out] 계산된 그라디언트 어텐션 가중치를 저장할 배열
 *                                     (shape: [batch_size, num_query, num_heads, 
 *                                              num_levels, num_point])
 *
 * @return                      없음 (커널 함수이므로 반환값 없음)
 *
 * @details
 *  - 이 함수는 backward pass에서 사용되는 col2im_coord 연산을 CUDA를 통해 GPU에서 수행합니다.
 *  - 각 스레드는 grad_sampling_loc과 grad_attn_weight의 특정 위치를 업데이트하며, bilinear 보간을 통해 그라디언트를 분배합니다.
 *  - `GET_BLOCKS(num_kernels)`와 `CUDA_NUM_THREADS`는 커널 실행 시 필요한 블록과 스레드 수를 계산합니다.
 *  - 커널 실행 후, CUDA 에러를 체크하여 문제 발생 시 에러 메시지를 출력합니다.
 *
 * @note
 *  - `GET_BLOCKS`와 `CUDA_NUM_THREADS`는 사전에 정의된 매크로로, GPU의 스레드 및 블록 설정을 조정합니다.
 *  - 이 함수는 Deformable DETR, Multi-Scale Deformable Attention 등에서 주로 사용됩니다.
 ******************************************************************************/
template <typename scalar_t>
void ms_deformable_col2im_coord_cuda(cudaStream_t stream,
                                     const scalar_t *data_col,
                                     const scalar_t *data_value,
                                     const int64_t *data_spatial_shapes,
                                     const int64_t *data_level_start_index,
                                     const scalar_t *data_sampling_loc,
                                     const scalar_t *data_attn_weight,
                                     const int batch_size,
                                     const int spatial_size,
                                     const int num_heads,
                                     const int channels,
                                     const int num_levels,
                                     const int num_query,
                                     const int num_point,
                                     scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight)
{
    //---------------------------------------------------------------------------
    // 커널 실행에 필요한 스레드 수 계산
    // num_kernels = batch_size * num_query * num_heads * num_levels * num_point * 2
    //---------------------------------------------------------------------------
    const int num_kernels = batch_size * num_query * num_heads * num_levels * num_point * 2;

    //---------------------------------------------------------------------------
    // ms_deformable_col2im_coord_gpu_kernel 커널 실행
    //---------------------------------------------------------------------------
    ms_deformable_col2im_coord_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
           0, stream>>>(
            num_kernels,
            data_col,
            data_value,
            data_spatial_shapes,
            data_level_start_index,
            data_sampling_loc,
            data_attn_weight,
            batch_size,
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            grad_sampling_loc, grad_attn_weight);

    //---------------------------------------------------------------------------
    // CUDA 에러 체크
    //---------------------------------------------------------------------------
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ms_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
    }
}
