package controllers

import javax.inject.Inject

import play.api.mvc.{AbstractController, ControllerComponents}
import services.ModelBuilder


class ModelController @Inject()(cc: ControllerComponents) extends AbstractController(cc) {

  /**
    * Create an Action to render an HTML page with a welcome message.
    * The configuration in the `routes` file means that this method
    * will be called when the application receives a `GET` request with
    * a path of `/`.
    */
  def buildModel = Action {
    ModelBuilder.buildModel()
    Ok("I've just built a new model for you...")
  }
}