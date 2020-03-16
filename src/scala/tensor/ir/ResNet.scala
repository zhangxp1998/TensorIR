package tensor.ir
import lms.macros.SourceContext

import scala.annotation.tailrec
import scala.util.continuations.{cps, reset}

object ResNet {

  def main(args: Array[String]) {
    val dslDriver = new TensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        trait Layer extends Diff {
          def forward(x: TensorR[Float]): TensorR[Float]@diff
          def parameters(): Seq[TensorR[Float]]
        }
        class Conv2D(val inChannels: Int, val outChannels: Int, val kernelSize: Int, val stride: Int, val padding: Int) extends Layer {
          val kernels: TensorR[Float] =
            TensorR.rand(Seq(outChannels, inChannels, kernelSize, kernelSize), AllocationType.Parameter)
          val bias: TensorR[Float] = TensorR.rand(Seq(outChannels), AllocationType.Parameter)
          def forward(x: TensorR[Float]): TensorR[Float]@diff = {
            assert(x.x.dims(1) == inChannels)
            x.conv2d(kernels, bias, padding, stride)
          }
          def parameters(): Seq[TensorR[Float]] = Seq(kernels, bias)
        }
        class BatchNorm(val inChannels: Int) extends Layer {
          val gamma_beta = TensorR.rand(Seq(2, inChannels), AllocationType.Parameter)
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = x.batchNorm(gamma_beta)

          override def parameters(): Seq[TensorR[Float]] = Seq(gamma_beta)
        }
        class ReLU() extends Layer {
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = x.relu()

          override def parameters(): Seq[TensorR[Float]] = Seq()
        }
        class FCLayer(val inSize: Int, val outSize: Int) extends Layer {
          val weight = TensorR.rand(Seq(inSize, outSize), AllocationType.Parameter)
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = x matmul weight

          override def parameters(): Seq[TensorR[Float]] = Seq(weight)
        }

        class Sequential(val layers: Layer*) extends Layer {
//          @tailrec
          final def sequential_forward(x: TensorR[Float], remain: List[Layer]): TensorR[Float]@diff = remain match {
            case head :: tail => sequential_forward(head.forward(x), tail)
            case Nil => x
          }
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = sequential_forward(x, layers.toList)
          lazy val params = layers.map(_.parameters()).reduce((a, b) => a ++ b)
          override def parameters(): Seq[TensorR[Float]] = params
        }

        class ResidualBlock(val inChannels: Int, val outChannels: Int, val stride: Int = 1) extends Layer {
          val left = new Sequential(
            new Conv2D(inChannels, outChannels, 3, stride, 1),
            new BatchNorm(outChannels),
            new ReLU(),
            new Conv2D(outChannels, outChannels, 3, 1, 1),
            new BatchNorm(outChannels)
          )
          val shortcut =
            if (inChannels == outChannels)
              new Sequential()
            else
              new Sequential(
                new Conv2D(inChannels, outChannels, 1, 2, 0),
                new BatchNorm(outChannels)
              )
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = {
            val out = left.forward(x)
            (out + shortcut.forward(x)).relu()
          }

          override def parameters(): Seq[TensorR[Float]] = left.parameters() ++ shortcut.parameters()
        }
        class ResNet extends Layer {
          val layer = new Sequential(
            new Conv2D(3, 8, 3, 1, 1),
            new BatchNorm(8),
            new ReLU(),
            new ResidualBlock(8, 16, 2),
          )

          override def forward(x: TensorR[Float]): TensorR[Float]@diff = layer.forward(x)

          override def parameters(): Seq[TensorR[Float]] = layer.parameters()
        }
        trait Optimizer {
          def step(): Unit
        }
        class GradientDescent(val layer: Layer, val learningRate: Float) extends Optimizer {
          override def step(): Unit = layer.parameters().foreach { l =>
            l.x -= l.d * Const(learningRate)
            println(l.x.unsafe_apply(0))
          }
        }

        val batchSize = 10
        val imgSize = 32
        val input = Tensor.rand(Seq(batchSize, 3, imgSize, imgSize), AllocationType.Data)
        val resNet = new ResNet()
        val optimizer = new GradientDescent(resNet, 0.01)

        def grad(f: TensorR[Float] => TensorR[Float]@cps[Unit])(x: Tensor[Float]) = {
          val z = new TensorR[Float](x, Tensor.zero[Float](x.dims, AllocationType.Gradient))
          reset({
            val res = f(z)
            res.d = Tensor.fill[Float](res.x.dims, 1, AllocationType.Gradient)
            println(res.x.unsafe_apply(0))
          })
          z.d
        }
        grad(x => resNet.forward(x))(input)
        optimizer.step()
        Const(())
      }
    }


    dslDriver.eval("5").foreach(println)
  }
}