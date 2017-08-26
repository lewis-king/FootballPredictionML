package services.helpers

import javax.inject.Singleton

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame

@Singleton
object FeatureAssembler {

  def constructFeaturesDataFrame(df: DataFrame): DataFrame = {

    val assembler = new VectorAssembler().setInputCols(Array("HomeTeamIndex",
      "AwayTeamIndex"/*, "FTHG"*//*, "FTAG"*//*, "FTRIndex"*/,
      "homeWinOdds", "drawOdds", "awayWinOdds")).setOutputCol("features")

    return assembler.transform(df)
  }

}
