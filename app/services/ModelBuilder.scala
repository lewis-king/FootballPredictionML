package services

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType

object ModelBuilder {

  def buildModel() = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    println("#################################")
    println("Lewis' Football Prediction Engine")
    println("#################################")


    //spark-shell --packages saurfang:spark-knn:0.1.1 --conf spark.sql.warehouse.dir=file:///c:/tmp/spark-warehouse

    val spark = SparkSession.builder()
      .master("local")
      .appName("Spark ML Football Predictor")
      .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    val files = Array[String]("resources/201617/championship_201617.csv", "resources/201617/leagueone_201617.csv",
      "resources/201617/leaguetwo_201617.csv")

    //Take CSV and transform into DataFrame
    val df_e1 = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .option("inferSchema", "true")
      .csv("resources/201617/premierleague_201617.csv")

    var df = df_e1.select("Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A")

    files.foreach(fileName => {
      val df_1 = spark.read
        .format("csv")
        .option("header", "true") //reading the headers
        .option("mode", "DROPMALFORMED")
        .option("inferSchema", "true")
        .csv(fileName)

      val df_view = df_1.select("Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A")

      df = df.union(df_view)
    })

    df.collect.foreach(println)

    //END CSV reading and DF creation

    //NOT USING ATM - COULD USE FOR IMPROVED MODEL
    //define the buckets/splits for odds
    val homeOddsSplits = Array(1.0,1.3,1.6,1.9,2.2,2.5,2.8,3.1,3.4,3.8,6.0,10.0,20.0,50.0,Double.PositiveInfinity)
    val drawOddsSplits = Array(1.0,1.3,1.6,1.9,2.2,2.5,2.8,3.1,3.4,3.8,6.0,10.0,20.0,50.0,Double.PositiveInfinity)
    val awayOddsSplits = Array(1.0,1.3,1.6,1.9,2.2,2.5,2.8,3.1,3.4,3.8,6.0,10.0,20.0,50.0,Double.PositiveInfinity)
    val homeOddsBucketize = new Bucketizer().setInputCol("B365H").setOutputCol("homeOddsBucketed").setSplits(homeOddsSplits)
    val drawOddsBucketize = new Bucketizer().setInputCol("B365D").setOutputCol("drawOddsBucketed").setSplits(drawOddsSplits)
    val awayOddsBucketize = new Bucketizer().setInputCol("B365A").setOutputCol("awayOddsBucketed").setSplits(awayOddsSplits)

    val newDF = homeOddsBucketize.transform(df).select("B365H","homeOddsBucketed")
    // END
    val homeTeamIndexer = new StringIndexer().setInputCol("HomeTeam").setOutputCol("HomeTeamIndex")
    val homeTeamIndexed = homeTeamIndexer.fit(df).transform(df)
    val stringIndexerModel = homeTeamIndexer.fit(df)
    stringIndexerModel.write.overwrite().save("target/model/stringIndexer")
    val awayTeamIndexer = stringIndexerModel.setInputCol("AwayTeam").setOutputCol("AwayTeamIndex")
    val awayTeamIndexed = stringIndexerModel.transform(homeTeamIndexed)

    val resultIndexer = new StringIndexer().setInputCol("FTR").setOutputCol("FTRIndex")
    val indexed = resultIndexer.fit(awayTeamIndexed).transform(awayTeamIndexed)

    val assembler = new VectorAssembler().setInputCols(Array("HomeTeamIndex",
      "AwayTeamIndex"/*, "FTHG"*//*, "FTAG"*//*, "FTRIndex"*/,
      "B365H", "B365D", "B365A")).setOutputCol("features")
    df = assembler.transform(indexed)

    import spark.implicits._

    //Set Home Team Goals to predict
    var df_home = df.select(df("FTHG").cast(DoubleType).as("label"), df("*"))

    df_home.show()

    //Set Away Team Goals to predict
    val df_away = df.select(df("FTAG").cast(DoubleType).as("label"), df("*"))

    //Split the data to give our training data and data to test against
    val splits = df_home.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    //split2
    val splits_2 = df_away.randomSplit(Array(0.7, 0.3))
    val (trainingData_2, testData_2) = (splits_2(0), splits_2(1))

    //  create the classifier,  set parameters for training**
    val regressor = new RandomForestRegressor().setImpurity("variance").setMaxDepth(5).setNumTrees(20)
      .setFeatureSubsetStrategy("auto").setSeed(5043).setMaxBins(120)

    //  use the random forest classifier  to train (fit) the model**
    val model = regressor.fit(trainingData)
    val model_2 = regressor.fit(trainingData_2)

    // run the  model on test features to get predictions**
    val predictions = model.transform(testData)
    val predictions_2 = model_2.transform(testData_2)
    // As you can see, the previous model transform produced a new columns: rawPrediction, probablity and prediction.**
    val result = predictions.select("HomeTeam", "AwayTeam", "FTHG", "FTAG", "prediction")
    val result_2 = predictions_2.select("HomeTeam", "AwayTeam", "FTHG", "FTAG", "prediction")
    result.show(50)
    result_2.show(50)

    model.write.overwrite().save("target/model/football_rf_home")
    model_2.write.overwrite().save("target/model/football_rf_away")
    //new PrintWriter("src/main/resources/tree_def/tree.txt") {write (model.toDebugString); close}
    println(model.toDebugString)
  }

}