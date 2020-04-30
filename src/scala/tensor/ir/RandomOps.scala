package tensor.ir

import lms.core.Backend.{Const, _}
import lms.core._
import lms.core.stub.Adapter.typeMap
import lms.core.stub._
import lms.macros.SourceContext

import scala.collection.mutable

trait RandomOps extends Base {
  val RAND = Backend.Const("random")
  def randInt(): Rep[Int] = {
    Wrap[Int](Adapter.g.reflectEffect("rand_int")(RAND)(RAND))
  }
  // Random number in range [0.0f, 1.0f]
  def randFloat(): Rep[Float] = {
    Wrap[Float](Adapter.g.reflectEffect("rand_float")(RAND)(RAND))
  }
  def sampleDistribution(dis: Rep[UniformFloatDistribution]): Rep[Float] = {
    Wrap[Float](Adapter.g.reflectEffect("rand_sample", Unwrap(dis))(RAND)(RAND))
  }
  def getUniformFloatDistribution(lower: Float, upper: Float): Rep[UniformFloatDistribution] = {
    Wrap[UniformFloatDistribution](Adapter.g.reflect("rand_create_distribution", Backend.Const((lower, upper))))
  }
  def randFloat(a: Float, b: Float): Rep[Float] = {
    val distribution = getUniformFloatDistribution(a, b)
    sampleDistribution(distribution)
  }
}

trait RandomOpsCodegen extends CGenBase {
  registerHeader("<random>")
  registerDatastructures("random_device"){
    emit(
      """
        |static std::random_device rd;
        |static const auto seed = rd();
        |thread_local std::mt19937 rng{seed};
        |std::uniform_int_distribution<> int_dis{};
        |std::uniform_real_distribution<> real_dis{0.0, 1.0};
        |""".stripMargin)
  }

  override def remap(m: Manifest[_]): String = m.toString() match {
    case s if s.contains("UniformFloatDistribution") => "std::uniform_real_distribution<float>"
    case _ => super.remap(m)
  }
  override def shallow(n: Node): Unit = n match {
    case Node(s, "rand_int", _, _) =>
      emit("int_dis(rng)")
    case Node(s, "rand_float", _, _) =>
      emit("real_dis(rng)")
    case Node(s, "rand_create_distribution", Const((a: Float, b: Float)):: Nil, _) =>
      emit(s"std::uniform_real_distribution<float>{$a, $b}")
    case Node(s, "rand_sample", dis::Nil, _) =>
      shallow(dis)
      emit("(rng)")
    case _ => super.shallow(n)
  }
}

// A type that maps to C++ std::uniform_real_distribution<> real_dis{a, b};
trait UniformFloatDistribution {
}