package services

import model.{Match, Prediction}
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.SparkConf
import play.api.libs.json.JsValue

import scala.collection._


object PredictionEngine {

    def predict(matchesJson: Option[JsValue]): List[Prediction] = {

        val sparkConf = new SparkConf()
        //sparkConf.setJars(Seq("target/scala-2.11/footballpredictionmlengine_2.11-1.0.jar"))

        val spark = SparkSession.builder()
          .master("local")
          .appName("Spark ML Football Predictor")
          .config(sparkConf)
          .getOrCreate()

        import spark.implicits._

        val rdd = spark.sparkContext.makeRDD(matchesJson.get.toString() :: Nil)
        var df = spark.read.json(rdd)
        //Load model

        val homeModel = RandomForestRegressionModel.load("target/model/football_rf_home")
        val awayModel = RandomForestRegressionModel.load("target/model/football_rf_away")

        val stringIndexerModel = StringIndexerModel.load("target/model/teamIndexer")
        val divisionIndexerModel = StringIndexerModel.load("target/model/divisionIndexer")

        val homeTeamIndexer = stringIndexerModel.setInputCol("homeTeam").setOutputCol("HomeTeamIndex").setHandleInvalid("keep") // options are "keep", "error" or "skip"
        val homeTeamIndexed = homeTeamIndexer.transform(df)
        val awayTeamIndexer = stringIndexerModel.setInputCol("awayTeam").setOutputCol("AwayTeamIndex").setHandleInvalid("keep") // options are "keep", "error" or "skip"
        val awayTeamIndexed = awayTeamIndexer.transform(homeTeamIndexed)
        val divisionIndexer = divisionIndexerModel.setInputCol("div").setOutputCol("DivIndex")
        val divisionIndexed = divisionIndexer.transform(awayTeamIndexed)
        //will need Full Time result index (FTR Index)

        val assembler = new VectorAssembler().setInputCols(Array("DivIndex",
            /*"HomeTeamIndex", "AwayTeamIndex",*/
            /*"HomeTeamOverallFormL3", "AwayTeamOverallFormL3", "HomeTeamHomeFormL3", "AwayTeamAwayFormL3",*/
            "HomeTeamPromoted", "AwayTeamPromoted",
            "HomeTeamAvgGoalsScoredOverall", "HomeTeamAvgGoalsConcededOverall", "AwayTeamAvgGoalsScoredOverall", "AwayTeamAvgGoalsConcededOverall", "HomeTeamAvgGoalsScoredHome", "HomeTeamAvgGoalsConcededHome", "AwayTeamAvgGoalsScoredAway", "AwayTeamAvgGoalsConcededAway",
            "HomeTeamAvgGoalsScoredOverallL3", "HomeTeamAvgGoalsConcededOverallL3", "AwayTeamAvgGoalsScoredOverallL3", "AwayTeamAvgGoalsConcededOverallL3", "HomeTeamAvgGoalsScoredHomeL3", "HomeTeamAvgGoalsConcededHomeL3", "AwayTeamAvgGoalsScoredAwayL3", "AwayTeamAvgGoalsConcededAwayL3"
            )).setOutputCol("features")
        df = assembler.transform(divisionIndexed)
        val df1 = df.withColumn("FTHG", functions.lit(0.0))
        val df2 = df1.withColumn("FTAG", functions.lit(0.0))
        val df3_home = df2.select(df2("FTHG").cast(DoubleType).as("label"), df2("*"))
        val df3_away = df2.select(df2("FTAG").cast(DoubleType).as("label"), df2("*"))

        val homeGoal_predictions = homeModel.transform(df3_home)
        val awayGoal_predictions = awayModel.transform(df3_away)

        homeGoal_predictions.show(false)
        awayGoal_predictions.show(false)

        val homeFilteredPre = homeGoal_predictions.select("homeTeam", "awayTeam", "prediction")
        val homeFiltered = homeFilteredPre.withColumnRenamed("prediction", "homeTeamScore")
        val awayFilteredPre = awayGoal_predictions.select("homeTeam", "prediction")
        val awayFiltered = awayFilteredPre.withColumnRenamed("prediction", "awayTeamScore")
        df = homeFiltered.join(awayFiltered, Seq("homeTeam"), joinType="outer")
        df.show(false)

        import org.apache.spark.sql.functions.round
        val homeScoreRounded = df.withColumn("homeTeamScore", round($"homeTeamScore", 2))
        val dfRounded = homeScoreRounded.withColumn("awayTeamScore", round($"awayTeamScore", 2))

        val predictions = mutable.MutableList[Prediction]()

        val predictionArr = dfRounded.collect()
        predictionArr.foreach(row => {
          val p = Prediction(row.getAs("homeTeam"), row.getAs("awayTeam"), row.getAs("homeTeamScore"), row.getAs("awayTeamScore"))
          predictions += p
        })

        return predictions.toList;
    }


}