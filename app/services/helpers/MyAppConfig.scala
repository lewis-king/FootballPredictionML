package services.helpers

import com.typesafe.config.ConfigFactory

object MyAppConfig {
  private val config =  ConfigFactory.load()

  object AWS {
    private val AWS = config.getConfig("aws")

    lazy val s3_bucket = AWS.getString("s3.bucket")
    lazy val s3_access_key = AWS.getString("s3.accesskey")
    lazy val s3_secret_key = AWS.getString("s3.secretkey")
  }
}