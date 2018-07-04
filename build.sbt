name := "FootballPredictionMLEngine"

version := "1.0"

scalaVersion := "2.11.8"

lazy val root = (project in file(".")).enablePlugins(PlayScala)
resolvers += Resolver.sonatypeRepo("snapshots")

val sparkVersion = "2.2.0"
val playVersion = "2.6.2"

dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-core" % "2.8.7"
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % "2.8.7"
dependencyOverrides += "com.fasterxml.jackson.module" % "jackson-module-scala_2.11" % "2.8.7"
dependencyOverrides += "com.google.guava" % "guava" % "16.0.1"

libraryDependencies += guice
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "com.typesafe" % "config" % "1.2.1"
libraryDependencies += "com.amazonaws" % "aws-java-sdk" % "1.3.11"
libraryDependencies += "org.apache.hadoop" % "hadoop-aws" % "2.6.0"