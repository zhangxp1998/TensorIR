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
  override def shallow(n: Node): Unit = n match {
    case Node(s, "rand_int", _, _) =>
      emit("int_dis(rng")
    case Node(s, "rand_float", _, _) =>
      emit("real_dis(rng)")
    case _ => super.shallow(n)
  }
}
