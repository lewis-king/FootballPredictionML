package services

import model.{Match, Prediction}
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.{SparkConf}
import play.api.libs.json.JsValue

import scala.collection._


object PredictionEngine {

    def predict(matchesJson: Option[JsValue]): List[Prediction] = {

        val sparkConf = new SparkConf()
        sparkConf.setJars(Seq("target/scala-2.11/footballpredictionmlengine_2.11-1.0.jar"))

        val spark = SparkSession.builder()
          .master("local")
          .appName("Spark ML Football Predictor")
          .config(sparkConf)
          .getOrCreate()

        import spark.implicits._

        val rdd = spark.sparkContext.makeRDD(matchesJson.get.toString() :: Nil)
        val df = spark.read.json(rdd)
        //Load model

        val homeModel = RandomForestRegressionModel.load("target/model/football_rf_home")
        val awayModel = RandomForestRegressionModel.load("target/model/football_rf_away")

        val stringIndexerModel = StringIndexerModel.load("target/model/stringIndexer")

        val homeTeamIndexer = stringIndexerModel.setInputCol("homeTeam").setOutputCol("HomeTeamIndex")
        val homeTeamIndexed = homeTeamIndexer.transform(df)
        val awayTeamIndexer = stringIndexerModel.setInputCol("awayTeam").setOutputCol("AwayTeamIndex")
        val awayTeamIndexed = awayTeamIndexer.transform(homeTeamIndexed)

        val assembler = new VectorAssembler().setInputCols(Array("HomeTeamIndex",
            "AwayTeamIndex"/*, "FTHG"*//*, "FTAG"*//*, "FTRIndex"*/,
            "homeWinOdds", "drawOdds", "awayWinOdds")).setOutputCol("features")
        val df3 = assembler.transform(awayTeamIndexed)
        val df3_home = df3.select(df3("FTHG").cast(DoubleType).as("label"), df3("*"))
        val df3_away = df3.select(df3("FTAG").cast(DoubleType).as("label"), df3("*"))

        val homeGoal_predictions = homeModel.transform(df3_home)
        val awayGoal_predictions = awayModel.transform(df3_away)

        homeGoal_predictions.show()
        awayGoal_predictions.show()

        val homeFilteredPre = homeGoal_predictions.select("homeTeam", "awayTeam", "prediction")
        val homeFiltered = homeFilteredPre.withColumnRenamed("prediction", "homeTeamScore")
        val awayFilteredPre = awayGoal_predictions.select("homeTeam", "prediction")
        val awayFiltered = awayFilteredPre.withColumnRenamed("prediction", "awayTeamScore")

        val finalDF = homeFiltered.join(awayFiltered, Seq("homeTeam"), joinType="outer")
        finalDF.show()

        val predictions = mutable.MutableList[Prediction]()

        val predictionArr = finalDF.collect()
        predictionArr.foreach(row => {
          val p = Prediction(row.getAs("homeTeam"), row.getAs("awayTeam"), row.getAs("homeTeamScore"), row.getAs("awayTeamScore"))
          predictions += p
        })

        return predictions.toList;
    }


}