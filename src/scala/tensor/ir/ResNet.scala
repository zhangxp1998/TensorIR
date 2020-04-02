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
          def zero_grad(): Unit = {
            parameters().foreach(u => u.d.fill(0))
          }
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
          val bias = TensorR.rand(Seq(outSize), AllocationType.Parameter)
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = x.matmul(weight, Some(bias))

          override def parameters(): Seq[TensorR[Float]] = Seq(weight, bias)
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

        class Flatten extends Layer {
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = {
            val N = x.d.dims.head
            x.reshape(Seq(N, x.d.dims.product/N))
          }

          override def parameters(): Seq[TensorR[Float]] = Seq()
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
            new Conv2D(1, 3, 3, 1, 1),
            new BatchNorm(3),
            new ReLU(),
            new Conv2D(3, 8, 3, 2, 1),
            new Flatten(),
            new FCLayer(1568, 10),
//            new ResidualBlock(3, 8, 2),
          )

          override def forward(x: TensorR[Float]): TensorR[Float]@diff = layer.forward(x)

          override def parameters(): Seq[TensorR[Float]] = layer.parameters()
        }
        trait Optimizer {
          def step(): Unit
        }
        class GradientDescent(val layer: Layer, val learningRate: Float) extends Optimizer {
          override def step(): Unit = layer.parameters().foreach { l =>
//              println(l.d.unsafe_apply(0))
              l.x -= l.d * Const(learningRate)
            }
        }

        val batchSize = 6000
        val imgSize = 28
        val input = Tensor[Float](Seq(batchSize, 1, imgSize, imgSize), AllocationType.Data)
        input.fread("train_images.bin", "uint8_t")
        val labels = Tensor[Int](Seq(batchSize), AllocationType.Data)
        labels.fread("train_labels.bin", "double")
        val resNet = new ResNet()
        val optimizer = new GradientDescent(resNet, 1e-4)

        def grad(f: TensorR[Float] => TensorR[Float]@cps[Unit])(x: Tensor[Float]) = {
          val z = new TensorR[Float](x, Tensor.zero[Float](x.dims, AllocationType.Gradient))
          reset({
            val res = f(z)
            res.d.fill(1.0f)
            println(res.x.unsafe_apply(0))
          })
          z.d
        }
        for (_ <- 0 until 100: Rep[Range]) {
//          println("===========Iteration Begin===========")
          resNet.zero_grad()
          grad(x => resNet.forward(x).softmaxLoss(labels))(input)
          optimizer.step()
        }
      }
    }


    dslDriver.eval("5").foreach(println)
  }
}