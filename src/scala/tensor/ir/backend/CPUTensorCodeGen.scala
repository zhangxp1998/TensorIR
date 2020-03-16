package scala.tensor.ir.backend

import java.io.PrintWriter

import lms.core.Backend.{Block, Const, Def, Exp, Node, Sym}
import lms.core.{Backend, Graph}
import lms.core.stub.DslGenC
import tensor.ir.{AllocationType, RandomOpsCodegen, StagedMemoryAllocator}
import tensor.ir.StagedMemoryAllocator.{Allocation, Deallocation, MemoryBlock}

import scala.collection.mutable

trait CPUTensorCodeGen extends DslGenC with RandomOpsCodegen {
  val debug = false
  override def init(g: Graph): Graph = {
    val graph = super.init(g)
    super.init(memoryPlanning(graph))
  }
  def saveMemoryRequests(requests: Seq[MemoryRequest]): Unit = {
    val writer = new PrintWriter("requests.csv")
    var totalMem = 0
    try {
      writer.println("begin,end,size,location,type")
      requests.sortBy(_.allocatedTime).foreach(request => {
        writer.println(s"${request.allocatedTime},${request.deallocatedTime},${request.size},$totalMem,${request.allocType.id}")
        totalMem += request.size
      })
    } finally {
      writer.close()
    }
  }
  def saveMemoryPlan(requests: Map[Sym, MemoryRequest], plan: Map[Int, MemoryBlock]): Unit = {
    val writer = new PrintWriter("plan.csv")
    try {
      writer.println("begin,end,size,location,type")
      requests.foreach{case (sym, request) =>
        writer.println(s"${request.allocatedTime},${request.deallocatedTime},${request.size},${plan(sym.n).begin},${request.allocType.id}")
      }
    } finally {
      writer.close()
    }
  }
  def allocateMemory(requests: mutable.HashMap[Sym, MemoryRequest]) = {
    val (perm, intermediate) = requests.partition{case (_, r) => r.allocType == AllocationType.Parameter || r.allocType == AllocationType.Gradient}
    val events = MemoryPlanningTraverser.toEvents(intermediate.toMap).values.toSeq
    val partialPlan = StagedMemoryAllocator.allocate(events)
    var memused = partialPlan.values.map(a => a.begin + a.size) match {
      case s if s.isEmpty => 0
      case s => s.max
    }
    perm.foreach{ case (sym, request) =>
      partialPlan(sym.n) = MemoryBlock(memused, request.size)
      memused += request.size
    }
    partialPlan
  }
  def memoryPlanning(g: Graph): Graph = {
    val traverser = new MemoryPlanningTraverser()
    traverser(g)
    if (debug) {
      val scale: Int => Int = a => Math.log(a).toInt
      val scaleRequest: MemoryRequest => MemoryRequest =
        req => new MemoryRequest(req.allocatedTime, req.deallocatedTime, req.lastUseSym, scale(req.size), req.src, req.isCopy, req.allocType)
      saveMemoryRequests(traverser.requests.values.map(scaleRequest).toSeq)
      val events = traverser.events.values

      val scaledEvents = events.map {
        case Allocation(id, size) => Allocation(id, scale(size))
        case Deallocation(id, size, sym) => Deallocation(id, scale(size), sym)
      }
      saveMemoryPlan(traverser.requests.map{
        case (sym, req) => sym ->
          scaleRequest(req)}.toMap,
        StagedMemoryAllocator.allocate(scaledEvents.toSeq).toMap)
    }

    val allocationPlan = allocateMemory(traverser.requests)

    val transformer = new CPUMemoryPlanningTransformer(allocationPlan.toMap, traverser.reusedSyms.toMap)
    val newGraph = transformer.transform(g)
    typeMap = transformer.newTypeMap
    newGraph
  }
  doRename = false
  //  val _shouldInline = shouldInline
  var totalMemory: Int = 0
  var allocationPlan: Map[Int, MemoryBlock] = Map()
  registerHeader("<string.h>", "<algorithm>")
  registerHeader("<cblas.h>", "<dnnl.hpp>")
  registerHeader("<sys/mman.h>", "<unistd.h>")
  registerLibrary("-L/opt/OpenBLAS/lib", "-I/opt/OpenBLAS/include", "-lopenblas", "-g")
  registerLibrary("-lmkldnn")
  registerDatastructures("heap") {
    emit("char *heap = NULL;")
  }
  registerDatastructures("epsilon") {
    emit("constexpr float epsilon = 1e-7f;")
  }
  registerDatastructures("eng"){
    emit("dnnl::engine eng{dnnl::engine::kind::cpu, 0};")
  }
  registerDatastructures("stream") {
    emit("dnnl::stream stream(eng);")
  }
  registerInit("heap_init") {
    emit("heap = (char*)get_mem(1024*1024*1024);")
  }
  registerTopLevelFunction("tensor_copy"){
    emit(
      """
        |static void *memdup(void* source, size_t bytes) {
        |   void *copy = malloc(bytes);
        |   memcpy(copy, source, bytes);
        |   return copy;
        |}
        |""".stripMargin)
  }
  registerTopLevelFunction("conv2d_forward") {
    emit(
      """
        |template
        |<size_t N, size_t C, size_t H, size_t W, size_t OutChannels, size_t KernelSize, size_t padding, size_t stride>
        |static dnnl::convolution_forward::primitive_desc get_conv2d_prim_desc(const dnnl::engine& eng) {
        |  using namespace dnnl;
        |  memory::dims conv2_src_tz = {N, C, H, W};
        |  memory::dims conv2_weights_tz = {OutChannels, C, KernelSize, KernelSize};
        |  memory::dims conv2_bias_tz = {OutChannels};
        |  memory::dims conv2_dst_tz = {N, OutChannels, static_cast<long long>((H+2*padding-KernelSize+1)/stride), static_cast<long long>((W+2*padding-KernelSize+1)/stride)};
        |  // create memory descriptors for convolution data w/ no specified format
        |  auto conv2_src_md = memory::desc({conv2_src_tz}, memory::data_type::f32, memory::format_tag::any);
        |  auto conv2_bias_md = memory::desc({conv2_bias_tz}, memory::data_type::f32, memory::format_tag::any);
        |  auto conv2_weights_md = memory::desc({conv2_weights_tz}, memory::data_type::f32, memory::format_tag::any);
        |  auto conv2_dst_md = memory::desc({conv2_dst_tz}, memory::data_type::f32, memory::format_tag::any);
        |  memory::dims conv2_strides = {stride, stride};
        |  memory::dims conv2_padding = {padding, padding};
        |
        |  // create a convolution
        |//  try {
        |      auto conv2_desc = convolution_forward::desc(prop_kind::forward_inference,
        |              algorithm::convolution_auto, conv2_src_md, conv2_weights_md,
        |              conv2_bias_md, conv2_dst_md, conv2_strides, conv2_padding,
        |              conv2_padding);
        |      return convolution_forward::primitive_desc(conv2_desc, eng);
        |//  } catch (dnnl::error &e) {
        |//      std::cout << "DNNL error caught: " << std::endl
        |//                << "\tStatus: " << dnnl_status2str(e.status) << std::endl
        |//                << "\tMessage: " << e.what() << std::endl;
        |//  }
        |}
        |template
        |<size_t N, size_t C, size_t H, size_t W, size_t OutChannels, size_t KernelSize, size_t padding, size_t stride>
        |static void conv2d_forward(const dnnl::engine& eng, dnnl::stream& stream, const dnnl::memory& input, const dnnl::memory& output, const dnnl::memory& weights, const dnnl::memory& bias) {
        |  using namespace dnnl;
        |  static convolution_forward::primitive_desc prim_desc = get_conv2d_prim_desc<N, C, H, W, OutChannels, KernelSize, padding, stride>(eng);
        |  static auto conv2 = convolution_forward(prim_desc);
        |  conv2.execute(stream, {{DNNL_ARG_SRC, input},
        |  {DNNL_ARG_WEIGHTS, weights},
        |  {DNNL_ARG_BIAS, bias},
        |    {DNNL_ARG_DST, output}});
        |}
        |""".stripMargin)
  }
  registerTopLevelFunction("batchnorm_forward") {
    emit(
      """
        |template
        |<size_t N, size_t C, size_t H, size_t W>
        |static inline dnnl::batch_normalization_forward::primitive_desc get_batchnorm_prim_desc(const dnnl::engine& engine) {
        |  using namespace dnnl;
        |  memory::dims src_dims = {N, C, H, W};
        |  auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        |  // Create operation descriptor.
        |  auto bnorm_d = batch_normalization_forward::desc(
        |          prop_kind::forward_training, src_md, epsilon,
        |          normalization_flags::use_scale_shift);
        |  auto bnorm_pd
        |          = batch_normalization_forward::primitive_desc(bnorm_d, engine);
        |  auto mean_desc = bnorm_pd.mean_desc();
        |  auto var_desc = bnorm_pd.variance_desc();
        |  auto weight_desc = bnorm_pd.weights_desc();
        |  assert(mean_desc.data.ndims == 1);
        |  assert(mean_desc.dims()[0] == C);
        |  assert(var_desc.data.ndims == 1);
        |  assert(var_desc.dims()[0] == C);
        |//  assert(workspace_desc.data.ndims == 0);
        |  assert(weight_desc.data.ndims == 2);
        |  assert(weight_desc.dims()[0] == 2);
        |  assert(weight_desc.dims()[1] == C);
        |//  assert(workspace_desc.dims()[0] == C*H*W);
        |  return bnorm_pd;
        |}
        |
        |template
        |<size_t N, size_t C, size_t H, size_t W>
        |static void batchnorm_forward(const dnnl::engine& eng, dnnl::stream& stream,
        |     const dnnl::memory& src, const dnnl::memory& avg, const dnnl::memory& variance,
        |     const dnnl::memory& scale_shift, const dnnl::memory& dst) {
        |  using namespace dnnl;
        |  static batch_normalization_forward::primitive_desc prim_desc = get_batchnorm_prim_desc<N, C, H, W>(eng);
        |  static auto batchnorm = batch_normalization_forward(prim_desc);
        |  assert(src.get_desc() == prim_desc.src_desc());
        |  assert(avg.get_desc() == prim_desc.mean_desc());
        |  assert(variance.get_desc() == prim_desc.variance_desc());
        |  assert(scale_shift.get_desc() == prim_desc.weights_desc());
        |  assert(dst.get_desc() == prim_desc.dst_desc());
        |  batchnorm.execute(stream, {
        |    {DNNL_ARG_SRC, src},
        |    {DNNL_ARG_MEAN, avg},
        |    {DNNL_ARG_VARIANCE, variance},
        |    {DNNL_ARG_SCALE_SHIFT, scale_shift},
        |    {DNNL_ARG_DST, dst}
        |  });
        |}
        |""".stripMargin)
  }
  registerTopLevelFunction("bump_allocate") {
    emit(
      """
        |static void *bump_allocate(size_t size) {
        |   void *old_heap = heap;
        |   heap += size;
        |   return old_heap;
        |}
        |""".stripMargin
    )
  }
  registerTopLevelFunction("get_mem") {
    emit(
      """
        |void *get_mem(size_t size) {
        |  size_t page_size = getpagesize();
        |  size = (size + page_size - 1) / page_size * page_size;
        |  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
        |                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        |  return p;
        |}
        |""".stripMargin)
  }

  val is_tensor_new_ops = Set("tensor-new", "heap-offset")

  override def remap(m: Manifest[_]): String = m.toString() match {
    case s if s.contains("MemDims") => "dnnl::memory::dims"
    case s if s.contains("MemDesc") => "dnnl::memory"
    case _ => super.remap(m)
  }

  override def shallow(node: Node): Unit = node match {
    case Node(s, "tensor-new", Const(manifest: Manifest[_])::Backend.Const(dims: Seq[Int])::Const(allocType)::Nil, eff) =>
      emit(s"((${remap(manifest)}*)malloc(${dims.product} * sizeof(${remap(manifest)})))")
    case Node(s, "heap-offset", Const(manifest: Manifest[_])::Const(blk: MemoryBlock)::src, eff) =>
      if (src.isEmpty)
        emit(s"((${remap(manifest)}*)(heap+${blk.begin} * sizeof(${remap(manifest)})))")
      else
        shallow(src.head)

    case Node(s, "tensor-apply", List(_, tensor, Const(idx: Seq[Int]), Const(dims: Seq[Int])), _) =>
      val sizes = dims.scanRight(1)(_ * _).tail
      shallow(tensor)
      emit(s"[${idx.zip(sizes).map{case (a, b) => a*b}.sum}]")
    case Node(s, "tensor-apply", List(_, tensor, idx), _) =>
      // Comming from unsafe_apply
      shallow(tensor)
      emit("[")
      shallow(idx)
      emit("]")
    case Node(s, "tensor-update", List(_, tensor, Const(idx: Seq[Int]), newVal, Const(dims: Seq[Int])), _) =>
      val sizes = dims.scanRight(1)(_ * _).tail
      shallow(tensor)
      emit(s"[${idx.zip(sizes).map{case (a, b) => a*b}.sum}] = ")
      shallow(newVal)
    case Node(s, "tensor-update", List(_, tensor, idx, newVal), _) =>
      shallow(tensor)
      emit("[")
      shallow(idx)
      emit("] = ")
      shallow(newVal)
    case Node(s, "tensor-fill", List(mA, tensor, fillVal, Const(dims: Seq[Int])), _) =>
      val totalSize = dims.product
      emit("std::fill(")
      shallow(tensor)
      emit(", ")
      shallow(tensor)
      emit(s" + $totalSize, ${quote(fillVal)})")


    case Node(s, "tensor-copy", List(mA, tensor, Const(dims: Seq[Int]), Const(allocType)), _) =>
      val manifest = mA match {case Const(mani: Manifest[_]) => mani}
      val totalSize = dims.product
      val byteSize = s"$totalSize * sizeof(${remap(manifest)})"
      emit(s"((${remap(manifest)}*)")
      emit("(memdup(")
      shallow(tensor)
      emit(", ")
      emit(byteSize.toString)
      emit(")))")
    case Node(s, "heap-offset-copy", Const(manifest: Manifest[_])::tensor::Const(blk: MemoryBlock)::Const(dims: Seq[Int])::_, eff) =>
      emit(s"((${remap(manifest)} *)memcpy(heap+${blk.begin} * sizeof(${remap(manifest)}), ")
      shallow(tensor)
      val byteSize = s"${dims.product} * sizeof(${remap(manifest)})"
      emit(s", $byteSize))")

    case Node(s, "matrix-multiply", List(mA, lhs, rhs, result, Const(Seq(m: Int, k: Int, n: Int))), _) =>
      if (mA.toString != "Float") {
        throw new RuntimeException(s"Only floating point values are supported: ${mA.toString}")
      }
      emit(s"cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, $m, $n, $k, 1, ")
      shallow(lhs)
      emit(s", $m, ")
      shallow(rhs)
      emit(s", $k, 0, ")
      shallow(result)
      emit(s", $m)")

    case n @ Node(s,"P",List(x),_) =>
      emit("""printf("""")
      emit(format(x))
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      shallow(x)
      emit(")")
    case Node(s, "tensor-accumulate-range", List(Const(mA: Manifest[_]), data, begin, end), _) =>
      emit("std::accumulate(")
      shallow(data)
      emit("+")
      shallow(begin)
      emit(", ")
      shallow(data)
      emit("+")
      shallow(end)
      emit(s", ${remap(mA)}(0))")
    case Node(s, "tensor-transform-range", List(Const(mA: Manifest[_]), data, block: Block, begin, end), _) =>
      assert(block.in.length == 1)
      emit("std::transform(")
      shallow(data)
      emit("+")
      shallow(begin)
      emit(", ")
      shallow(data)
      emit("+")
      shallow(end)
      emit(", ")
      shallow(data)
      emit(s", [&](${remap(mA)} ")
      shallow(block.in.head)
      emit(")")
      quoteBlockPReturn(traverse(block))
      emit(")")

    case Node(s, "tensor-transform-index", List(Const(mA: Manifest[_]), data, block: Block, Const(dims: Seq[Int])), _) =>
      assert(block.in.length == 1)
      val totalSize = dims.product
      val counter = s"i${s.n}"
      emit(s"for(size_t $counter = 0; $counter < $totalSize; ++$counter) {")
      emit(s"const size_t ")
      shallow(block.in.head)
      emit(s" = $counter;")
      shallow(data)
      emit("[")
      shallow(block.in.head)
      emit("] = ")
      quoteBlockP(traverse(block))
      emit(";}")
    case Node(s, "tensor-convolution", List(mA, data, kernel, output, Const(dims: Seq[Int]), Const(kernelDims: Seq[Int])), _) =>
      // TODO implement convolution, this is just a stub
      emit("/*Stub for tensor convolution TODO implement this*/")
    case Node(s, "batchnorm-forward", List(Const(dims: Seq[Int]), Const(epsilon: Float), src, avg, variance, gamma_beta, dst), _) =>
      // TODO support custom epsilon
      val Seq(n, c, h, w) = dims
      emit(s"batchnorm_forward<$n, $c, $h, $w>(eng, stream, ")
      shallow(src)
      emit(", ")
      shallow(avg)
      emit(", ")
      shallow(variance)
      emit(", ")
      shallow(gamma_beta)
      emit(", ")
      shallow(dst)
      emit(")")
    case Node(s, "max", List(lhs, rhs), _) =>
      emit(s"std::max<${remap(typeMap(s))}>(")
      shallow(lhs)
      emit(", ")
      shallow(rhs)
      emit(")")
    case Node(s, "tensor-convolution2d", List(mA, input, output, kernels, bias, Const(Seq(n, c, h, w)), Const(Seq(oc, kh, padding, stride))), _) =>
      emit(s"conv2d_forward<$n, $c, $h, $w, $oc, $kh, $padding, $stride>(eng, stream, ")
      shallow(input)
      emit(", ")
      shallow(output)
      emit(", ")
      shallow(kernels)
      emit(", ")
      shallow(bias)
      emit(")")
    case Node(s, "mem-dims", List(Backend.Const(dims: Seq[Int])), _) =>
      emit(s"dnnl::memory::dims({${ dims.mkString(", ")} })")
    case Node(s, "mem-desc", List(memDims, data, Const(dims: Seq[Int])), _) =>
      emit("dnnl::memory({")
      shallow(memDims)
      val format = dims.length match {
        case 1 => "a"
        case 2 => "ab"
        case 3 => "abc"
        case 4 => "nchw"
      }
      emit(s", dnnl::memory::data_type::f32, dnnl::memory::format_tag::$format}, eng, ")
      shallow(data)
      emit(")")
    case _ => super.shallow(node)
  }
  def format(x: Def): String = x match {
    case exp: Exp => exp match {
      case s@Sym(_) => typeMap(s).toString match {
        case m if m.matches("""Array\[.*\]""") =>
          "%p"
        case "Float" =>
          "%f"
        case "Int" =>
          "%d"
      }
      case Const(_: Int) => "%d"
      case Const(_: Float) => "%f"
      case _ => "%s"
    }
    case _ => "%s"
  }
}

