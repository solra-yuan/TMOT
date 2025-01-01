#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

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


template <typename scalar_t>
__device__ scalar_t ms_deform_attn_get_gradient_weight(scalar_t h, scalar_t w,
                                                       const int gh, const int gw, const int height, const int width)
{
  if (h <= -1 || h >= height || w <= -1 || w >= width)
  {
    // empty
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t weight = 0;
  if (gh == h_low && gw == w_low)
    weight = (gh + 1 - h) * (gw + 1 - w);
  if (gh == h_low && gw == w_high)
    weight = (gh + 1 - h) * (w + 1 - gw);
  if (gh == h_high && gw == w_low)
    weight = (h + 1 - gh) * (gw + 1 - w);
  if (gh == h_high && gw == w_high)
    weight = (h + 1 - gh) * (w + 1 - gw);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_get_coordinate_weight(scalar_t h, scalar_t w, const int m, const int c,
                                                         const int height, const int width, const int nheads, const int channels,
                                                         const scalar_t *bottom_data, const int bp_dir)
{
  if (h <= -1 || h >= height || w <= -1 || w >= width)
  {
    // empty
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    int ptr1 = h_low * width * nheads * channels + w_low * nheads * channels + m * channels + c;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    int ptr2 = h_low * width * nheads * channels + w_high * nheads * channels + m * channels + c;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    int ptr3 = h_high * width * nheads * channels + w_low * nheads * channels + m * channels + c;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    int ptr4 = h_high * width * nheads * channels + w_high * nheads * channels + m * channels + c;
    v4 = bottom_data[ptr4];
  }

  if (bp_dir == 1)
  {
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
    if (h_low >= 0 && w_low >= 0)
      weight += -1 * (h_low + 1 - h) * v1;
    if (h_low >= 0 && w_high <= width - 1)
      weight += (h_low + 1 - h) * v2;
    if (h_high <= height - 1 && w_low >= 0)
      weight += -1 * (h - h_low) * v3;
    if (h_high <= height - 1 && w_high <= width - 1)
      weight += (h - h_low) * v4;
  }

  return weight;
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
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
  // launch batch_size * num_levels * num_query * num_point * channels cores
  // data_value: batch_size, spatial_size, num_heads, channels
  // data_sampling_loc: batch_size, num_query, num_heads, num_levels, num_point, 2
  // data_attn_weight: batch_size, num_query, num_heads, num_levels, num_point
  // data_col: num_levels*num_point, batch_size, num_query, num_heads, channels
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int c_col = index % channels;
    const int p_col = (index / channels) % num_point;
    const int q_col = (index / channels / num_point) % num_query;
    const int l_col = (index / channels / num_point / num_query) % num_levels;
    const int b_col = index / channels / num_point / num_query / num_levels;
    const int level_start_id = data_level_start_index[l_col];
    const int spatial_h = data_spatial_shapes[l_col * 2];
    const int spatial_w = data_spatial_shapes[l_col * 2 + 1];

    // num_heads, channels
    scalar_t *data_col_ptr = data_col 
                           + (  c_col 
                              + channels * 0
                              + channels * num_heads * q_col
                              + channels * num_heads * num_query * b_col
                              + channels * num_heads * num_query * batch_size * p_col
                              + channels * num_heads * num_query * batch_size * num_point * l_col);
    // spatial_h, spatial_w, num_heads, channels
    const scalar_t *data_value_ptr = data_value 
                                   + (b_col * spatial_size * num_heads * channels + level_start_id * num_heads * channels);  
    // num_heads, num_levels, num_point, 2
    const scalar_t *data_sampling_loc_ptr = data_sampling_loc 
                                          + (  b_col * num_query * num_heads * num_levels * num_point * 2
                                             + q_col * num_heads * num_levels * num_point * 2);
    // num_heads, num_levels, num_point
    const scalar_t *data_attn_weight_ptr = data_attn_weight 
                                         + (  b_col * num_query * num_heads * num_levels * num_point
                                            + q_col * num_heads * num_levels * num_point);

    for (int i = 0; i < num_heads; ++i)
    {
      const int data_loc_h_ptr = i * num_levels * num_point * 2 + l_col * num_point * 2 + p_col * 2 + 1;
      const int data_loc_w_ptr = i * num_levels * num_point * 2 + l_col * num_point * 2 + p_col * 2;
      const int data_weight_ptr = i * num_levels * num_point + l_col * num_point + p_col;
      const scalar_t loc_h = data_sampling_loc_ptr[data_loc_h_ptr];
      const scalar_t loc_w = data_sampling_loc_ptr[data_loc_w_ptr];
      const scalar_t weight = data_attn_weight_ptr[data_weight_ptr];
      scalar_t val = static_cast<scalar_t>(0);
      const scalar_t h_im = loc_h * spatial_h - 0.5;
      const scalar_t w_im = loc_w * spatial_w - 0.5;
      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
      {
        val = ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, i, c_col);
      }
      *data_col_ptr = val * weight;
      data_col_ptr += channels;
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel(const int n,
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
  // launch batch_size * num_levels * num_query * num_point * num_heads * channels cores
  // grad_value: batch_size, spatial_size, num_heads, channels
  // data_sampling_loc: batch_size, num_query, num_heads, num_levels, num_point, 2
  // data_attn_weight: batch_size, num_query, num_heads, num_levels, num_point
  // data_col: batch_size, num_query, num_heads, channels
  CUDA_KERNEL_LOOP(index, n)
  {
    const int c_col = index % channels;
    const int m_col = (index / channels) % num_heads;
    const int p_col = (index / channels / num_heads) % num_point;
    const int q_col = (index / channels / num_heads / num_point) % num_query;
    const int l_col = (index / channels / num_heads / num_point / num_query) % num_levels;
    const int b_col = index / channels / num_heads / num_point / num_query / num_levels;
    const int level_start_id = data_level_start_index[l_col];
    const int spatial_h = data_spatial_shapes[l_col * 2];
    const int spatial_w = data_spatial_shapes[l_col * 2 + 1];

      const scalar_t col = data_col[  c_col
                                  + channels * m_col
                                  + channels * num_heads * q_col
                                  + channels * num_heads * num_query * b_col];
    int sampling_ptr = b_col * num_query * num_heads * num_levels * num_point
                    + q_col * num_heads * num_levels * num_point
                    + m_col * num_levels * num_point
                    + l_col * num_point
                    + p_col;
    const scalar_t sampling_x = data_sampling_loc[2 * sampling_ptr] * spatial_w - 0.5;
    const scalar_t sampling_y = data_sampling_loc[2 * sampling_ptr + 1] * spatial_h - 0.5;
    const scalar_t attn_weight = data_attn_weight[sampling_ptr];
    const scalar_t cur_top_grad = col * attn_weight;
    const int cur_h = (int)sampling_y;
    const int cur_w = (int)sampling_x;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < spatial_h &&
            cur_w + dx >= 0 && cur_w + dx < spatial_w &&
            abs(sampling_y - (cur_h + dy)) < 1 &&
            abs(sampling_x - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = b_col * spatial_size * num_heads * channels 
                                  + (level_start_id + (cur_h+dy)*spatial_w + (cur_w+dx)) * num_heads * channels 
                                  + m_col * channels
                                  + c_col;
          scalar_t weight = ms_deform_attn_get_gradient_weight(sampling_y, sampling_x, cur_h + dy, cur_w + dx, spatial_h, spatial_w);
          atomicAdd(grad_value + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_coord_gpu_kernel(const int n,
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
  // sampling_loc: batch_size, num_query, num_heads, num_levels, num_point, 2
  // attn_weight:  batch_size, num_query, num_heads, num_levels, num_point
  // column: batch_size, num_query, num_heads, channels
  // value: batch_size, spatial_size, num_heads, channels
  // num_kernels = batch_size * num_query * num_heads * num_levels * num_point * 2
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = 0, wval = 0;

    const int loc_c = index % 2;
    const int k = (index / 2) % num_point;
    const int l = (index / 2 / num_point) % num_levels;
    const int m = (index / 2 / num_point / num_levels) % num_heads;
    const int q = (index / 2 / num_point / num_levels / num_heads) % num_query;
    const int b = index / 2 / num_point / num_levels / num_heads / num_query;
    const int level_start_id = data_level_start_index[l];
    const int spatial_h = data_spatial_shapes[l * 2];
    const int spatial_w = data_spatial_shapes[l * 2 + 1];

    const scalar_t *data_col_ptr = data_col 
                                 +( m * channels
                                  + q * channels * num_heads
                                  + b * channels * num_heads * num_query);
    const scalar_t *data_value_ptr = data_value 
                                   + (  0 * channels  
                                      + level_start_id * channels * num_heads
                                      + b * channels * num_heads * spatial_size);
    scalar_t sampling_x = data_sampling_loc[(index / 2) * 2] * spatial_w - 0.5;
    scalar_t sampling_y = data_sampling_loc[(index / 2) * 2 + 1] * spatial_h - 0.5;
    const scalar_t attn_weight = data_attn_weight[index / 2];

    for (int col_c = 0; col_c < channels; col_c += 1)
    {
      const scalar_t col = data_col_ptr[col_c];
      if (sampling_x <= -1 || sampling_y <= -1 || sampling_x >= spatial_w || sampling_y >= spatial_h)
      {
        sampling_x = sampling_y = -2;
      }
      else
      {
        wval += col * ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, sampling_y, sampling_x, m, col_c);
      }
      const scalar_t weight = ms_deform_attn_get_coordinate_weight(
          sampling_y, sampling_x, m, col_c,
          spatial_h, spatial_w, num_heads, channels,
          data_value_ptr, loc_c);
      val += weight * col * attn_weight;
    }
    if (loc_c == 0)
      val *= spatial_w;
    else if (loc_c == 1)
      val *= spatial_h;
    grad_sampling_loc[index] = val;
    if (loc_c % 2 == 0)
      grad_attn_weight[index / 2] = wval;
  }
}

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
  // num_axes should be smaller than block size
  const int num_kernels = batch_size * num_levels * num_query * num_point * channels;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
         0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight,
          batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

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
  const int num_kernels = batch_size * num_levels * num_query * num_point * num_heads * channels;
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
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

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
  // data_sampling_loc: batch_size, num_query, num_heads, num_levels, num_point, 2
  // data_attn_weight: batch_size, num_query, num_heads, num_levels, num_point
  const int num_kernels = batch_size * num_query * num_heads * num_levels * num_point * 2;
  ms_deformable_col2im_coord_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
         0, stream>>>(num_kernels,
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
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}