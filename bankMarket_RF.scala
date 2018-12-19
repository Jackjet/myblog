import org.apache.spark.sql.SparkSession

object RandomForestClassifierBank {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RandomForestClassifierExample").master("local[2]")
      .getOrCreate()
    // $example on$
    // Load and parse the data file, converting it to a DataFrame.
    val df = spark.read.format("csv").option("header","true").option("inferSchema",true.toString).load("D:\\code\\sparkProject\\sparkInput\\bank-additional/bank-additional-full_washed.csv")
    // Split the data into training and test sets (30% held out for testing).
    df.show(3)
    df.printSchema

    val colNames = df.columns


    val featureCols = Array("age","education","default","housing","loan","duration","campaign","pdays","previous",
      "emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed","poutcome_failure",
      "poutcome_nonexistent","poutcome_success","job_admin","job_blue-collar","job_entrepreneur",
      "job_housemaid","job_management","job_retired","job_self-employed","job_services","job_student","job_technician",
      "job_unemployed","marital_divorced","marital_married","marital_single","contact_cellular","contact_telephone",
      "month_apr","month_aug","month_dec","month_jul","month_jun","month_mar","month_may","month_nov","month_oct",
      "month_sep","day_of_week_fri","day_of_week_mon","day_of_week_thu","day_of_week_tue","day_of_week_wed" )

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val classifier = new RandomForestClassifier()
      .setLabelCol("y")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("y")

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(25, 31))
      .addGrid(classifier.maxDepth, Array(5, 10))
      .addGrid(classifier.numTrees, Array(20, 60))
      .addGrid(classifier.impurity, Array("entropy", "gini"))
      .build()

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array( classifier))

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // Train model. This also runs the indexers.
    val model = cv.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)
    // Select example rows to display.
    predictions.select("*").show(10)


    val accuracy = evaluator.evaluate(predictions)
    println("accuracy fitting" + accuracy)


//    val rfModel = model.stages(0).asInstanceOf[RandomForestClassificationModel]
//    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
//    // $example off$

    spark.stop()
  }
}
