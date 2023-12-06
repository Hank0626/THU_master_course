import java.util.Date
import org.apache.spark.SparkContext

// 读取 Word Count 数据
val words = sc.textFile("/dsjxtjc/2023214278/wc_dataset_2GB.txt")

// 使用flatMap, map 和 reduceByKey的方法
val start = new Date().getTime()
val result1 = words.flatMap(l => l.split(" ")).map(w => (w, 1)).reduceByKey(_ + _)
result1.first()
val end = new Date().getTime
val duration1 = (end - start) / 1000.0  // 以秒为单位

// 使用flatMap 和 countByValue
val start = new Date().getTime()
val result2 = words.flatMap(l => l.split(" ")).countByValue()
val end = new Date().getTime
val duration2 = (end - start) / 1000.0  // 以秒为单位

// 判断两个结果是否相等
val areEqual = result1.collect().toMap == result2

println(s"用flatMap, map 和 reduceByKey的方法的时间: $duration1 秒")
println(s"用flatMap 和 countByValuey的方法的时间: $duration2 秒")
println(s"两种方式的结果是否相等: $areEqual")