#ifndef __TENSOR_DIFF_H
#define __TENSOR_DIFF_H
#include <dnnl.hpp>
#include "tensor.h"
void matmul_backprop(const float *m1, const float *m2, const float *y,
                     float *d1, float *d2, const size_t M, const size_t K,
                     const size_t N) {
  // m1: M*K, m2: K*N, y: M*N
  // d1 += y * m2.T => M*N x N*K = M*K
  // d2 += m1.T * y => K*M x M*N = K*N
  dnnl_sgemm('N', 'T', M, N, K, 1.0f, y, M, m2, K, 1.0f, d1, M);
  dnnl_sgemm('T', 'N', K, M, N, 1.0f, m1, M, y, M, 1.0f, d2, M);
}
template <size_t N, size_t C, size_t H, size_t W>
static inline dnnl::batch_normalization_backward::primitive_desc
get_batchnorm_backward_prim_desc(const dnnl::engine &eng) {
  using namespace dnnl;
  memory::dims src_dims = {N, C, H, W};
  auto src_md =
      memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
  auto bnorm_d = batch_normalization_backward::desc{
      prop_kind::backward, src_md, src_md, EPSILON,
      normalization_flags::use_scale_shift};
  auto forward_pd = get_batchnorm_prim_desc<N, C, H, W>(eng);
  batch_normalization_backward::primitive_desc prim_desc{bnorm_d, eng,
                                                         forward_pd};
  return prim_desc;
}

template <size_t N, size_t C, size_t H, size_t W>
static void batchnorm_backward(const dnnl::engine &eng, dnnl::stream &stream,
                               const dnnl::memory &src, const dnnl::memory &diff_src,
                               const dnnl::memory &diff_dst, const dnnl::memory &avg,
                               const dnnl::memory &variance, const dnnl::memory &gamma_beta,
                               const dnnl::memory &diff_gamma_beta) {
  using namespace dnnl;
  static batch_normalization_backward::primitive_desc prim_desc =
      get_batchnorm_backward_prim_desc<N, C, H, W>(eng);
  static auto batchnorm = batch_normalization_backward(prim_desc);
  batchnorm.execute(stream, {/*Input**/
                             {DNNL_ARG_DIFF_DST, diff_dst},
                             {DNNL_ARG_SRC, src},
                             {DNNL_ARG_MEAN, avg},
                             {DNNL_ARG_VARIANCE, variance},
                             {DNNL_ARG_SCALE_SHIFT, gamma_beta},
                             /*Output**/
                             {DNNL_ARG_DIFF_SRC, diff_src},
                             {DNNL_ARG_DIFF_SCALE_SHIFT, diff_gamma_beta}});
}
template <size_t N, size_t C, size_t H, size_t W, size_t OC, size_t KernelSize,
          size_t padding, size_t stride>
static inline dnnl::convolution_backward_weights::primitive_desc
get_convolution_backward_prim_desc(const dnnl::engine &eng) {
  using namespace dnnl;
  memory::dims src_dims = {N, C, H, W};
  memory::dims conv_dst_tz = {N, OC,
                              (H + 2 * padding - KernelSize + 1) / stride,
                              (W + 2 * padding - KernelSize + 1) / stride};
  auto src_md =
      memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
  auto conv_weights_md =
      memory::desc({OC, C, KernelSize, KernelSize}, memory::data_type::f32,
                   memory::format_tag::nchw);
  auto conv_bias_md =
      memory::desc({OC}, memory::data_type::f32, memory::format_tag::a);
  auto conv_dst_md = memory::desc(conv_dst_tz, memory::data_type::f32,
                                  memory::format_tag::nchw);
  memory::dims conv_strides = {stride, stride};
  memory::dims conv_padding = {padding, padding};
  auto conv_pd =
      get_conv2d_prim_desc<N, C, H, W, OC, KernelSize, padding, stride>(eng);
  auto conv_bwd_weights_desc = convolution_backward_weights::desc(
      algorithm::convolution_direct, src_md, conv_weights_md, conv_bias_md,
      conv_dst_md, conv_strides, conv_padding, conv_padding);
  auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
      conv_bwd_weights_desc, eng, conv_pd);
  return conv_bwd_weights_pd;
}

template <size_t N, size_t C, size_t H, size_t W, size_t OC, size_t KernelSize,
          size_t padding, size_t stride>
static void convolution_backward(const dnnl::engine &eng, dnnl::stream &stream,
                                 const dnnl::memory &diff_dst, const dnnl::memory &src,
                                 const dnnl::memory &diff_weights,
                                 const dnnl::memory &diff_bias) {
  using namespace dnnl;
  static convolution_backward_weights::primitive_desc prim_desc =
      get_convolution_backward_prim_desc<N, C, H, W, OC, KernelSize, padding,
                                         stride>(eng);
  static auto conv = convolution_backward_weights(prim_desc);
  assert(prim_desc.diff_bias_desc() == diff_bias.get_desc());
  assert(prim_desc.diff_dst_desc() == diff_dst.get_desc());
  assert(prim_desc.diff_weights_desc() == diff_weights.get_desc());
  assert(prim_desc.src_desc() == src.get_desc());
  conv.execute(stream, {{DNNL_ARG_SRC, src},
                        {DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_DIFF_BIAS, diff_bias},
                        {DNNL_ARG_DIFF_WEIGHTS, diff_weights}});
}
#endif
