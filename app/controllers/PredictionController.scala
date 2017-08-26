package controllers

import javax.inject.Inject

import model.{Match, Prediction}
import play.api.libs.json.{JsValue, Json}
import play.api.mvc._
import services.PredictionEngine

/**
  * This controller creates an `Action` to handle HTTP requests to the
  * application's home page.
  */
class PredictionController @Inject()(cc: ControllerComponents) extends AbstractController(cc) {

  /**
    * Create an Action to render an HTML page with a welcome message.
    * The configuration in the `routes` file means that this method
    * will be called when the application receives a `GET` request with
    * a path of `/`.
    */
  def predictMatchOutcome = Action { request: Request[AnyContent] =>
    val body: AnyContent = request.body
    val jsonBody: Option[JsValue] = body.asJson

    // Expecting json body
    jsonBody.map { json =>
      val predictions = PredictionEngine.predict(jsonBody)
      implicit val predictionFormat = Json.format[Prediction]
      val jsonResponse = Json.obj("predictions" -> predictions)
      Ok(jsonResponse)
    }.getOrElse {
      BadRequest("Expecting application/json request body")
    }
  }
}