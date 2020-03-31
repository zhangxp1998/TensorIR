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
    val lastBlk = partialPlan.values.maxBy(b => b.begin + b.size)
    println(s"Memory Usage: ${lastBlk.begin + lastBlk.size}")
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
  registerHeader("\"tensor.h\"")
  registerHeader("<sys/mman.h>", "<unistd.h>")
  registerDatastructures("heap") {
    emit("char *heap = NULL;")
  }
  registerDatastructures("eng"){
    emit("dnnl::engine eng{dnnl::engine::kind::cpu, 0};")
  }
  registerDatastructures("stream") {
    emit("dnnl::stream stream(eng);")
  }
  registerInit("heap_init") {
    emit("heap = (char*)get_mem(1024UL*1024*1024*8);")
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
  registerTopLevelFunction("get_mem") {
    emit(
      """
        |void *get_mem(size_t size) {
        |  size_t page_size = getpagesize();
        |  size = (size + page_size - 1) / page_size * page_size;
        |  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
        |                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        |  if (p == NULL) {
        |     perror("mmap() failed");
        |  }
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
    case Node(s, "heap-offset", Const(manifest: Manifest[_])::Const(blk: MemoryBlock)::src, _) =>
      emit(s"/*${blk.toString}*/")
      if (src.isEmpty)
        emit(s"((${remap(manifest)}*)(heap+${blk.begin}))")
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
      emit(s"((${remap(manifest)} *)memcpy(heap+${blk.begin}, ")
      shallow(tensor)
      val byteSize = s"${dims.product} * sizeof(${remap(manifest)})"
      emit(s", $byteSize))")

    case Node(s, "matrix-multiply", List(mA, lhs, rhs, result, Const(Seq(m: Int, k: Int, n: Int))), _) =>
      if (mA.toString != "Float") {
        throw new RuntimeException(s"Only floating point values are supported: ${mA.toString}")
      }
      emit(s"sgemm('N', 'N', ")
      shallow(lhs)
      emit(", ")
      shallow(rhs)
      emit(", ")
      shallow(result)
      emit(s", $m, $k, $n, 1.0f, 0.0f)")

    case n @ Node(s,"P",List(x),_) =>
      emit("""printf("""")
      emit(format(x))
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      shallow(x)
      emit(")")
    case Node(s, "tensor-accumulate-range", List(Const(mA: Manifest[_]), data, begin, end), _) =>
      emit("std::accumulate(")
      emitBeginEnd(data, begin, end)
      emit(s", ${remap(mA)}(0))")
    case Node(s, "tensor-transform-range", List(Const(mA: Manifest[_]), data, block: Block, begin, end), _) =>
      assert(block.in.length == 1)
      emit("std::transform(")
      emitBeginEnd(data, begin, end)
      emit(", ")
      shallow(data)
      emit("+")
      shallow(begin)
      emit(s", [&](${remap(mA)} ")
      shallow(block.in.head)
      emit(")")
      quoteBlockPReturn(traverse(block))
      emit(")")
    case Node(s, "tensor-foreach", List(Const(mA: Manifest[_]), data, block: Block, begin, end), _) =>
      assert(block.in.length == 1)
      emit("std::foreach(")
      emitBeginEnd(data, begin, end)
      emit(s", [&](${remap(mA)} ")
      shallow(block.in.head)
      emit(")")
      quoteBlock(traverse(block))
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
    case Node(s, "logsoftmax-forward", List(src, dst, Const((rows, rowSize))), _) =>
      emit(s"logsoftmax_forward<$rows, $rowSize>(eng, stream, ")
      shallow(src)
      emit(", ")
      shallow(dst)
      emit(")")
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
    case Node(s, "tensor-max", List(data, begin, end), _) =>
      emit(s"(*std::max_element(")
      shallow(data)
      emit(s"+")
      shallow(begin)
      emit(",")
      shallow(data)
      emit("+")
      shallow(end)
      emit("))")
    case Node(s, "exp", List(x), _) =>
      emit("std::exp(")
      shallow(x)
      emit(")")
    case Node(s, "log", List(x), _) =>
      emit("std::log(")
      shallow(x)
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
    case Node(s, "tensor-fread", List(Const(mA: Manifest[_]), data, Const(path: String), Const(dims: Seq[Int]), Const(dtype: String)), _) =>
      emit(s"load_bin_convert<$dtype, ${remap(mA)}>(")
      shallow(data)
      emit(s", ${quote(path)}, ${dims.product})")
    case Node(s, "tensor-mmap", List(Const(mA: Manifest[_]), Const(dims: Seq[Int]), Const(path: String)), _) =>
      val elem_count = dims.product
      emit(s"mmap_file<${remap(mA)}>(${quote(path)}, $elem_count)")
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
    case Node(s, "tensor-data-copy", List(mA, src, dst, Const(dims: Seq[Int])), _) =>
      val elemCount = dims.product
      emit("std::copy(")
      shallow(src)
      emit(", ")
      shallow(src)
      emit(s"+$elemCount, ")
      shallow(dst)
      emit(")")
    case Node(s, "tensor-fill", List(mA, data, fillVal, Const(dims: Seq[Int])), _) =>
      emit("std::fill(")
      emitBeginEnd(data, Const(0), Const(dims.product))
      emit(", ")
      shallow(fillVal)
      emit(")")
    case _ => super.shallow(node)
  }
  def emitBeginEnd(data: Def, begin: Def, end: Def): Unit = {
    shallow(data)
    if (begin != Const(0)) {
      emit("+")
      shallow(begin)
    }
    emit(", ")
    shallow(data)
    emit("+")
    shallow(end)
  }
  def quote(s: String): String = {
    "\"" + s.replaceAllLiterally("\\", "\\\\") + "\""
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

