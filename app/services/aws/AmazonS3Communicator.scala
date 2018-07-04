package services.aws

import java.io.ByteArrayInputStream

import com.amazonaws.ClientConfiguration
import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.services.s3.AmazonS3Client
import com.amazonaws.services.s3.model.ObjectMetadata
import org.slf4j.LoggerFactory
import services.helpers.MyAppConfig

object AmazonS3Communicator {
  val logger = LoggerFactory.getLogger(this.getClass().getName())
  val credentials = new BasicAWSCredentials(MyAppConfig.AWS.s3_access_key, MyAppConfig.AWS.s3_secret_key)
  val amazonS3Client = new AmazonS3Client(credentials)
  val client = new ClientConfiguration()
  client.setSocketTimeout(300000)
  /**
    * Upload a file to standard bucket on S3
    */
  def upload(meta: ObjectMetadata, stream: ByteArrayInputStream, filename: String): Boolean = {
    try {
      amazonS3Client.putObject(MyAppConfig.AWS.s3_bucket, filename, stream, meta); true
    } catch {
      case ex: Exception => logger.error(ex.getMessage(), ex); false
    }
  }
  /**
    * Deletes a file to standard bucket on S3
    */

  def delete(fileKeyName: String): Boolean = {
    try {
      amazonS3Client.deleteObject(MyAppConfig.AWS.s3_bucket, fileKeyName); true
    } catch {
      case ex: Exception => logger.error(ex.getMessage(), ex); false
    }
  }
  /**
    * Checks if the file exists on the  standard bucket of S3
    */

  def doesFileExist(fileKeyName: String): Boolean = {
    try {
      amazonS3Client.getObjectMetadata(MyAppConfig.AWS.s3_bucket, fileKeyName); true
    } catch {
      case ex: Exception => logger.error(ex.getMessage(), ex); false
    }
  }
}