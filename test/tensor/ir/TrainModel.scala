package tensor.ir

import java.io.{ByteArrayOutputStream, IOException}
import java.nio.ByteBuffer
import java.util

trait TrainModel extends TensorOps {

  object LoadImgAndLable {
    private val MAGIC_OFFSET = 0
    private val OFFSET_SIZE = 4 //in bytes

    private val LABEL_MAGIC = 2049
    private val IMAGE_MAGIC = 2051
    private val NUMBER_ITEMS_OFFSET = 4
    private val ITEMS_SIZE = 4
    private val NUMBER_OF_ROWS_OFFSET = 8
    private val ROWS_SIZE = 4
    val ROWS = 28
    private val NUMBER_OF_COLUMNS_OFFSET = 12
    private val COLUMNS_SIZE = 4
    val COLUMNS = 28
    private val IMAGE_OFFSET = 16
    private val IMAGE_SIZE = ROWS * COLUMNS
    private val IMAGE_NUM = 5000
  }

  class LoadImgAndLable(var labelFileName: String, var imageFileName: String) {
    @throws[IOException]
    def loadImages: Array[Tensor[Float]] = {
      val imgData = Tensor[Float](Seq(LoadImgAndLable.IMAGE_NUM, LoadImgAndLable.ROWS, LoadImgAndLable.COLUMNS), AllocationType.Data)
      val labelData = Tensor[Float](Seq(LoadImgAndLable.IMAGE_NUM, 1), AllocationType.Data)
      val labelBuffer = new ByteArrayOutputStream
      val imageBuffer = new ByteArrayOutputStream
      val labelInputStream = this.getClass.getResourceAsStream(labelFileName)
      val imageInputStream = this.getClass.getResourceAsStream(imageFileName)
      var read = 0
      val buffer = new Array[Byte](16384)
      while ( {
        (read = labelInputStream.read(buffer, 0, buffer.length)) != -1
      }) labelBuffer.write(buffer, 0, read)
      labelBuffer.flush()
      while ( {
        (read = imageInputStream.read(buffer, 0, buffer.length)) != -1
      }) imageBuffer.write(buffer, 0, read)
      imageBuffer.flush()
      val labelBytes = labelBuffer.toByteArray
      val imageBytes = imageBuffer.toByteArray
      val labelMagic = util.Arrays.copyOfRange(labelBytes, 0, LoadImgAndLable.OFFSET_SIZE)
      val imageMagic = util.Arrays.copyOfRange(imageBytes, 0, LoadImgAndLable.OFFSET_SIZE)

      if (ByteBuffer.wrap(labelMagic).getInt != LoadImgAndLable.LABEL_MAGIC)
        throw new IOException("Bad magic number in label file!")
      if (ByteBuffer.wrap(imageMagic).getInt != LoadImgAndLable.IMAGE_MAGIC)
        throw new IOException("Bad magic number in image file!")

      val numberOfLabels = ByteBuffer.wrap(util.Arrays.copyOfRange(labelBytes, LoadImgAndLable.NUMBER_ITEMS_OFFSET, LoadImgAndLable.NUMBER_ITEMS_OFFSET + LoadImgAndLable.ITEMS_SIZE)).getInt
      val numberOfImages = ByteBuffer.wrap(util.Arrays.copyOfRange(imageBytes, LoadImgAndLable.NUMBER_ITEMS_OFFSET, LoadImgAndLable.NUMBER_ITEMS_OFFSET + LoadImgAndLable.ITEMS_SIZE)).getInt
      if (numberOfImages != numberOfLabels)
        throw new IOException("The number of labels and images do not match!")

      val numRows = ByteBuffer.wrap(util.Arrays.copyOfRange(imageBytes, LoadImgAndLable.NUMBER_OF_ROWS_OFFSET, LoadImgAndLable.NUMBER_OF_ROWS_OFFSET + LoadImgAndLable.ROWS_SIZE)).getInt
      val numCols = ByteBuffer.wrap(util.Arrays.copyOfRange(imageBytes, LoadImgAndLable.NUMBER_OF_COLUMNS_OFFSET, LoadImgAndLable.NUMBER_OF_COLUMNS_OFFSET + LoadImgAndLable.COLUMNS_SIZE)).getInt
      if (numRows != LoadImgAndLable.ROWS && numRows != LoadImgAndLable.COLUMNS)
        throw new IOException("Bad image. Rows and columns do not equal " + LoadImgAndLable.ROWS + "x" + LoadImgAndLable.COLUMNS)

      for (i <- 0 until LoadImgAndLable.IMAGE_NUM) {
        val label = labelBytes(LoadImgAndLable.OFFSET_SIZE + LoadImgAndLable.ITEMS_SIZE + i)
        labelData(Seq(i, i)) = label
        for (j <- 0 until LoadImgAndLable.ROWS) {
          for (k <- 0 until LoadImgAndLable.COLUMNS) {
            imgData(Seq(i, j, k)) = imageBytes(LoadImgAndLable.IMAGE_OFFSET + i * LoadImgAndLable.IMAGE_SIZE + j + k)
          }
        }
      }
      var imgBuffer = new Array[Tensor[Float]](2)
      imgBuffer(0) = labelData
      imgBuffer(1) = imgData
      imgBuffer
    }
  }

}
