package tensor.ir

import org.scalatest.FunSuite

class ScalarDifferentiationTest extends FunSuite {
  test("add") {
    val grad = NumR.grad(a => a + a)(_: Double)
    for (x <- -10 until 10) {
      assert(grad(x) == 2)
    }
  }
  test("sub") {
    val grad = NumR.grad(a => 0 - a)(_: Double)
    for (x <- -10 until 10) {
      assert(grad(x) == -1)
    }
  }
  test("mul") {
    val grad = (x: Double, y: Double) => NumR.grad((a, b) => a * b)(x, y)
    for (x <- -10 until 10) {
      for (y <- -10 until 10) {
        assert(grad(x, y) == (y, x))
      }
    }
  }
  test("div") {
    val grad = (x: Double, y: Double) => NumR.grad((a, b) => a / b)(x, y)
    for (x <- -10 until 10) {
      for (y <- -10 until 10) {
        if (y != 0) {
          assert(grad(x, y) == (1.0/y, -x.toDouble/(y*y)))
        }
      }
    }
  }

  test("Composite") {
    val grad = NumR.grad(a => 2*a + a*a*a)(_: Double)
    for (x <- -10 until 10) {
      assert(grad(x) == 2 + 3*x*x)
    }
  }
}
