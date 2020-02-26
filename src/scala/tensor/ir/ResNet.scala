package tensor.ir

object ResNet {

  def main(args: Array[String]) {
    val dslDriver = new TensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        trait Layer extends Diff {
          def forward(x: TensorR[Float]): TensorR[Float]@diff
          def parameters(): Seq[TensorR[Float]]
        }
        class Conv2D(val inChannels: Int, val outChannels: Int, val kernelSize: Int, val stride: Int, val padding: Int) extends Layer {
          val kernels: Seq[TensorR[Float]] =
            scala.collection.immutable.Range(0, outChannels).map(_ => TensorR.rand(Seq(inChannels, kernelSize, kernelSize)))
          def forward(x: TensorR[Float]): TensorR[Float]@diff = {
            assert(x.x.dims(1) == inChannels)
            x.conv2d(kernels, padding, stride)
          }
          def parameters(): Seq[TensorR[Float]] = kernels
        }
        class BatchNorm(val inChannels: Int) extends Layer {
          val beta = TensorR.rand(Seq(inChannels))
          val gamma = TensorR.rand(Seq(inChannels))
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = x.batchNorm(gamma, beta)

          override def parameters(): Seq[TensorR[Float]] = Seq(beta, gamma)
        }
        class ReLU() extends Layer {
          override def forward(x: TensorR[Float]): TensorR[Float]@diff = x.relu()

          override def parameters(): Seq[TensorR[Float]] = Seq()
        }
        class FCLayer(val inSize: Int, val outSize: Int) extends Layer {
          val weight = TensorR.rand(Seq(inSize, outSize))
          override def forward(x: TensorR[Float]): TensorR[Float] = x matmul weight

          override def parameters(): Seq[TensorR[Float]] = Seq(weight)
        }

        class Sequential(val layers: Layer*) extends Layer {
          override def forward(x: TensorR[Float]): TensorR[Float] =
            layers.reduceLeft[TensorR[Float]]((tensor, layer) => layer.forward(tensor))
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
                new Conv2D(inChannels, outChannels, 1, 1, 0),
                new BatchNorm(outChannels)
              )
          override def forward(x: TensorR[Float]): TensorR[Float] = {
            val out = left.forward(x)
            (out + shortcut.forward(x)).relu()
          }

          override def parameters(): Seq[TensorR[Float]] = left.parameters() ++ shortcut.parameters()
        }
        class ResNet extends Layer {
          override def forward(x: TensorR[Float]): TensorR[Float] = ???

          override def parameters(): Seq[TensorR[Float]] = ???
        }
      }
    }


    dslDriver.eval("5").foreach(println)
  }
}