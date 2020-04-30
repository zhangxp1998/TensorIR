package tensor.ir

import lms.core.stub.{Adapter, Base}

trait Printf extends Base {
  def println(x: Rep[Any]*): Unit =
    Adapter.g.reflectWrite("P", x.map(a => Unwrap(a)): _*)(Adapter.CTRL)
}
