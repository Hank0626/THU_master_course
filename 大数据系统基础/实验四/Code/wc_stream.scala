import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._

val ssc = new StreamingContext(sc, Seconds(5))
ssc.checkpoint("/dsjxtjc/2023214278/wc_output_stream")

val lines = ssc.socketTextStream("thumm01", 14278)
val words = lines.flatMap(_.split(" "))

val wordCounts = words.map(word => (word, 1)).reduceByKey(_ + _)
val runningCounts = wordCounts.updateStateByKey((newCounts: Seq[Int], state: Option[Int]) => {
  Some(state.getOrElse(0) + newCounts.sum)
})

// 确保 runningCounts 是一个 Pair RDD，类型为 RDD[(String, Int)]
runningCounts.transform(rdd => rdd.sortByKey())
.saveAsTextFiles("/dsjxtjc/2023214278/wc_output_stream.txt")

runningCounts.print()

ssc.start()
ssc.awaitTermination()