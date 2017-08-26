package services.helpers

import javax.inject.Singleton

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.DataFrame

/**
  * Transforms a given DataFrame by indexing the column passed in
  */
@Singleton
object TeamIndexer {
  def teamIndexer(df: DataFrame, column: String): DataFrame = {
    val indexer = new StringIndexer().setInputCol(column).setOutputCol(column + "Index")
    return indexer.fit(df).transform(df)
  }

}
