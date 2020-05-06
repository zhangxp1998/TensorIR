#ifndef __TENSOR_DIFF_H
#define __TENSOR_DIFF_H
#include "tensor.h"
#include <dnnl.hpp>

void matmul_backprop(const float *m1, const float *m2, const float *y,
                     float *d1, float *d2, const size_t M, const size_t K,
                     const size_t N);

template <size_t N, size_t C, size_t H, size_t W>
dnnl::batch_normalization_backward::primitive_desc
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
void batchnorm_backward(const dnnl::engine &eng, dnnl::stream &stream,
                        const dnnl::memory &src, const dnnl::memory &diff_src,
                        const dnnl::memory &diff_dst, const dnnl::memory &avg,
                        const dnnl::memory &variance,
                        const dnnl::memory &gamma_beta,
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
dnnl::convolution_backward_weights::primitive_desc
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
void convolution_backward(const dnnl::engine &eng, dnnl::stream &stream,
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

template <size_t N, size_t C, size_t H, size_t W, size_t OC, size_t KernelSize,
size_t padding, size_t stride>
dnnl::convolution_backward_data::primitive_desc get_convolution_backward_data_prim_desc(const dnnl::engine& eng) {
  using namespace dnnl;
  memory::dims conv2_src_tz = {N, C, H, W};
  memory::dims conv2_weights_tz = {OC, C, KernelSize, KernelSize};
  memory::dims conv2_dst_tz = {
      N, OC,
      static_cast<long long>((H + 2 * padding - KernelSize + 1) / stride),
      static_cast<long long>((W + 2 * padding - KernelSize + 1) / stride)};
  // create memory descriptors for convolution data w/ no specified format
  auto conv2_src_md = memory::desc({conv2_src_tz}, memory::data_type::f32,
                                   memory::format_tag::any);
  auto conv2_weights_md = memory::desc(
      {conv2_weights_tz}, memory::data_type::f32, memory::format_tag::abcd);
  auto conv2_dst_md = memory::desc({conv2_dst_tz}, memory::data_type::f32,
                                   memory::format_tag::abcd);
  memory::dims conv2_strides = {stride, stride};
  memory::dims conv2_padding = {padding, padding};
  auto desc = convolution_backward_data::desc(algorithm::convolution_auto, conv2_src_md, conv2_weights_md, conv2_dst_md, conv2_strides, conv2_padding, conv2_padding);
  return convolution_backward_data::primitive_desc(desc, eng, get_conv2d_prim_desc<N, C, H, W, OC, KernelSize, padding, stride>(eng));
}

template <size_t N, size_t C, size_t H, size_t W, size_t OC, size_t KernelSize,
size_t padding, size_t stride>
void convolution_backward_data(const dnnl::engine& engine, dnnl::stream& stream, const dnnl::memory& diff_dst, const dnnl::memory& weights, const dnnl::memory& diff_src) {
  using namespace dnnl;
  static auto prim_desc = get_convolution_backward_data_prim_desc<N, C, H, W, OC, KernelSize, padding, stride>(engine);
  static auto conv = dnnl::convolution_backward_data(prim_desc);
  assert(prim_desc.diff_dst_desc() == diff_dst.get_desc());
  assert(prim_desc.weights_desc() == weights.get_desc());
  assert(prim_desc.diff_src_desc() == diff_src.get_desc());
  conv.execute(stream, {
    {DNNL_ARG_DIFF_DST, diff_dst},
    {DNNL_ARG_WEIGHTS, weights},
    {DNNL_ARG_DIFF_SRC, diff_src}
  });
}

template <size_t N, size_t IC>
dnnl::logsoftmax_backward::primitive_desc
get_logsoftmax_backward_prim_desc(const dnnl::engine &engine) {
  using namespace dnnl;
  memory::dims src_dims = {N, IC};
  auto src_md =
      memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ab);
  auto desc = logsoftmax_backward::desc(src_md, src_md, 1);
  return logsoftmax_backward::primitive_desc(
      desc, engine, get_logsoftmax_forward_prim_desc<N, IC>(engine));
}

template <size_t N, size_t IC>
void logsoftmax_backward(const dnnl::engine &engine, dnnl::stream &stream,
                         const dnnl::memory &diff_dst, const dnnl::memory &dst,
                         const dnnl::memory &diff_src) {
  using namespace dnnl;
  static auto prim_desc = get_logsoftmax_backward_prim_desc<N, IC>(engine);
  static auto logsoftmax = dnnl::logsoftmax_backward(prim_desc);
  assert(prim_desc.dst_desc() == dst.get_desc());
  assert(prim_desc.diff_dst_desc() == diff_dst.get_desc());
  assert(prim_desc.diff_src_desc() == diff_src.get_desc());
  logsoftmax.execute(stream, {
                                 {DNNL_ARG_DIFF_DST, diff_dst},
                                 {DNNL_ARG_DST, dst},
                                 {DNNL_ARG_DIFF_SRC, diff_src},
                             });
}

template <size_t N, size_t IC, typename T, typename Idx>
void nll_loss_backward(const T *diff_dst, const Idx *label, T *diff_src) {
  const auto gradient = - (*diff_dst)/N;
  for (size_t i = 0; i < N; i ++) {
      diff_src[i * IC + label[i]] = gradient;
  }
}

#endif
