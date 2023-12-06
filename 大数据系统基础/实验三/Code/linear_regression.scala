import scala.collection.mutable.ArrayBuffer
import scala.io.Source

// 用于矩阵运算的 Matrix 类
class Matrix(private val data: Array[Double], val rows: Int, val cols: Int) {
    require(data.length == rows * cols, "数据大小必须与行数 x 列数相匹配.")
    private val matrix = Array.tabulate(rows, cols) { (i, j) =>
        data(i * cols + j)
    }

    override def toString: String = matrix.map(_.mkString(" ")).mkString("\n")

    // 使得矩阵类m可以直接通过m(i, j)来访问第i行第j列的值
    def apply(row: Int, col: Int): Double = matrix(row - 1)(col - 1)

    // 矩阵 x 矩阵
    def *(that: Matrix): Matrix = {
        require(this.cols == that.rows, "前矩阵的列数和后矩阵的行数必须一致.")
        val newData = for {
            i <- 0 until this.rows
            j <- 0 until that.cols
        } yield (0 until this.cols).map(k => this.matrix(i)(k) * that.matrix(k)(j)).sum
        new Matrix(newData.toArray, this.rows, that.cols)
    }

    // 常数 x 矩阵
    def *(scalar: Double): Matrix = new Matrix(this.data.map(_ * scalar), this.rows, this.cols)

    // 矩阵 + 矩阵
    def +(that: Matrix): Matrix = {
        require(this.rows == that.rows && this.cols == that.cols, "矩阵维度必须一致.")
        new Matrix((this.data zip that.data).map { case (x, y) => x + y }, this.rows, this.cols)
    }

    // 矩阵 - 矩阵
    def -(that: Matrix): Matrix = {
        require(this.rows == that.rows && this.cols == that.cols, "矩阵维度必须一致.")
        new Matrix((this.data zip that.data).map { case (x, y) => x - y }, this.rows, this.cols)
    }

    // 矩阵转制
    def transpose: Matrix = {
        val transposedData = (0 until cols).flatMap(j => (0 until rows).map(i => matrix(i)(j)))
        new Matrix(transposedData.toArray, cols, rows)
    }

    // Squared difference sum
    def squaredDifferenceSum(that: Matrix): Double = {
        require(this.rows == that.rows && this.cols == that.cols, "矩阵维度必须一致.")
        (this.data zip that.data).map { case (x, y) => val diff = x - y; diff * diff }.sum
    }
}

// 用向量表示的多元线性回归object
object LinearRegression {
    def main(data_path: String, lr : Double = 0.001, max_iter : Int = 200): Unit = {
        val (x, y) = Source.fromFile(data_path).getLines().foldLeft((ArrayBuffer[Double](), ArrayBuffer[Double]())) {
        case ((xAcc, yAcc), line) =>
            val parts = line.split("\\s+") // 使用空格进行分割
            val yValue = parts.last.toDouble
            val xValues = 1.0 +: parts.init.map(_.toDouble) // 将最后一个值作为y，其余作为x
            (xAcc ++= xValues, yAcc += yValue)
        }
        val cols = (x.length / y.length).toInt
        val X_matrix = new Matrix(x.toArray, y.length, cols)
        val Y_matrix = new Matrix(y.toArray, y.length, 1)
        var w = new Matrix(Array.fill(cols)(1.0), cols, 1)

        for (i <- 1 until max_iter + 1) {
            val gradient = (X_matrix.transpose * (X_matrix * w - Y_matrix)) * (lr * 2)
            w -= gradient
            val J = ((X_matrix * w - Y_matrix).transpose * (X_matrix * w - Y_matrix)) * (1.0 / (2 * X_matrix.rows))
            if (i % 10 == 0) {
                println(s"迭代 $i: $J")
            }
            if (i % 50 == 0) {
                val mse = Y_matrix.squaredDifferenceSum(X_matrix * w) / Y_matrix.rows
                println(s"训练集上的均方误差: $mse")
            }
        }
        println(s"最终权重:\n$w")
    }
}
