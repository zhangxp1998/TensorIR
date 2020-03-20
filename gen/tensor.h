#ifndef __TENSOR_H
#define __TENSOR_H
#include <dnnl.hpp>
#include "tensor_constants.h"
template <size_t N, size_t C, size_t H, size_t W, size_t OutChannels,
          size_t KernelSize, size_t padding, size_t stride>
static dnnl::convolution_forward::primitive_desc
get_conv2d_prim_desc(const dnnl::engine &eng) {
  using namespace dnnl;
  memory::dims conv2_src_tz = {N, C, H, W};
  memory::dims conv2_weights_tz = {OutChannels, C, KernelSize, KernelSize};
  memory::dims conv2_bias_tz = {OutChannels};
  memory::dims conv2_dst_tz = {
      N, OutChannels,
      static_cast<long long>((H + 2 * padding - KernelSize + 1) / stride),
      static_cast<long long>((W + 2 * padding - KernelSize + 1) / stride)};
  // create memory descriptors for convolution data w/ no specified format
  auto conv2_src_md = memory::desc({conv2_src_tz}, memory::data_type::f32,
                                   memory::format_tag::any);
  auto conv2_bias_md = memory::desc({conv2_bias_tz}, memory::data_type::f32,
                                    memory::format_tag::any);
  auto conv2_weights_md = memory::desc(
      {conv2_weights_tz}, memory::data_type::f32, memory::format_tag::any);
  auto conv2_dst_md = memory::desc({conv2_dst_tz}, memory::data_type::f32,
                                   memory::format_tag::any);
  memory::dims conv2_strides = {stride, stride};
  memory::dims conv2_padding = {padding, padding};

  // create a convolution
  //  try {
  auto conv2_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_auto, conv2_src_md,
      conv2_weights_md, conv2_bias_md, conv2_dst_md, conv2_strides,
      conv2_padding, conv2_padding);
  return convolution_forward::primitive_desc(conv2_desc, eng);
  //  } catch (dnnl::error &e) {
  //      std::cout << "DNNL error caught: " << std::endl
  //                << "\tStatus: " << dnnl_status2str(e.status) << std::endl
  //                << "\tMessage: " << e.what() << std::endl;
  //  }
}
template <size_t N, size_t C, size_t H, size_t W, size_t OutChannels,
          size_t KernelSize, size_t padding, size_t stride>
void
conv2d_forward(const dnnl::engine &eng, dnnl::stream &stream,
               const dnnl::memory &input, const dnnl::memory &output,
               const dnnl::memory &weights, const dnnl::memory &bias) {
  using namespace dnnl;
  static convolution_forward::primitive_desc prim_desc =
      get_conv2d_prim_desc<N, C, H, W, OutChannels, KernelSize, padding,
                           stride>(eng);
  static auto conv2 = convolution_forward(prim_desc);
  conv2.execute(stream, {{DNNL_ARG_SRC, input},
                         {DNNL_ARG_WEIGHTS, weights},
                         {DNNL_ARG_BIAS, bias},
                         {DNNL_ARG_DST, output}});
}

template <size_t N, size_t C, size_t H, size_t W>
dnnl::batch_normalization_forward::primitive_desc
get_batchnorm_prim_desc(const dnnl::engine &engine) {
  using namespace dnnl;
  memory::dims src_dims = {N, C, H, W};
  auto src_md =
      memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
  // Create operation descriptor.
  auto bnorm_d = batch_normalization_forward::desc(
      prop_kind::forward_training, src_md, EPSILON,
      normalization_flags::use_scale_shift);
  auto bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_d, engine);
  auto mean_desc = bnorm_pd.mean_desc();
  auto var_desc = bnorm_pd.variance_desc();
  auto weight_desc = bnorm_pd.weights_desc();
  assert(mean_desc.data.ndims == 1);
  assert(mean_desc.dims()[0] == C);
  assert(var_desc.data.ndims == 1);
  assert(var_desc.dims()[0] == C);
  //  assert(workspace_desc.data.ndims == 0);
  assert(weight_desc.data.ndims == 2);
  assert(weight_desc.dims()[0] == 2);
  assert(weight_desc.dims()[1] == C);
  //  assert(workspace_desc.dims()[0] == C*H*W);
  return bnorm_pd;
}

template <size_t N, size_t C, size_t H, size_t W>
void batchnorm_forward(const dnnl::engine &eng, dnnl::stream &stream,
                              const dnnl::memory &src, const dnnl::memory &avg,
                              const dnnl::memory &variance,
                              const dnnl::memory &scale_shift,
                              const dnnl::memory &dst) {
  using namespace dnnl;
  static batch_normalization_forward::primitive_desc prim_desc =
      get_batchnorm_prim_desc<N, C, H, W>(eng);
  static auto batchnorm = batch_normalization_forward(prim_desc);
  assert(src.get_desc() == prim_desc.src_desc());
  assert(avg.get_desc() == prim_desc.mean_desc());
  assert(variance.get_desc() == prim_desc.variance_desc());
  assert(scale_shift.get_desc() == prim_desc.weights_desc());
  assert(dst.get_desc() == prim_desc.dst_desc());
  batchnorm.execute(stream, {{DNNL_ARG_SRC, src},
                             {DNNL_ARG_MEAN, avg},
                             {DNNL_ARG_VARIANCE, variance},
                             {DNNL_ARG_SCALE_SHIFT, scale_shift},
                             {DNNL_ARG_DST, dst}});
}

void load_file(char *data, const char *path, size_t size);
#endif
