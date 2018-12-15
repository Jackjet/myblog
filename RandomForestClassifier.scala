package com.spark.ML

/**
  * @ Author     ：maoyeqin
  * @ Date       ：Created in 15:32 2018/12/15
  * @ Description：${description}
  * @ Modified By：
  *
  * @Version: $version$
  */
import org.apache.spark._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

object Iris {

  case class Iris(
                   sepal_length: Float,
                   sepal_width: Float, petal_length: Float, petal_width: Float, Iris_class: Float
                 )

  def parseCredit(line: Array[Float]): Iris = {
    Iris(
      line(0),
      line(1) - 1, line(2), line(3), line(4)
    )
  }

  def parseRDD(rdd: RDD[String]): RDD[Array[Float]] = {
    rdd.map(_.split(",")).map(_.map(_.toFloat))
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("SparkDFebay").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val df = parseRDD(sc.textFile("D:\\code\\sparkProject\\sparkInput\\Iris_data.txt")).map(parseCredit).toDF().cache()

    val featureCols = Array("sepal_length", "sepal_width", "petal_length", "petal_width", "Iris_class")




    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
    df2.show

    val labelIndexer = new StringIndexer().setInputCol("Iris_class").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    df3.show
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), 5000)

    val classifier = new RandomForestClassifier()
    val model = classifier.fit(trainingData)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(25, 31))
      .addGrid(classifier.maxDepth, Array(5, 10))
      .addGrid(classifier.numTrees, Array(20, 60))
      .addGrid(classifier.impurity, Array("entropy", "gini"))
      .build()

    val steps: Array[PipelineStage] = Array(classifier)
    val pipeline = new Pipeline().setStages(steps)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val pipelineFittedModel = cv.fit(trainingData)

    val predictions = pipelineFittedModel.transform(testData)
    val accuracy = evaluator.evaluate(predictions)
    println("accuracy before pipeline fitting" + accuracy)

    println(pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))
  }
}
