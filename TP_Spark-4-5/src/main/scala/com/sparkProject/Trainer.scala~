package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** CHARGER LE DATASET **/

    val df: DataFrame = spark
      .read
      .option("header", "true")        // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column (ne sert pas à grand chose ici, car il met en string et retraiter au e))
      .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
      .parquet("/Users/christophethibault/IdeaProjects/TP_ParisTech_2017_2018_starter/prepared_trainingset")


    //df.show()
    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")
    //df.printSchema()

    /** TF-IDF **/
    //premier stage : tokens
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //deuxieme stage : StopWordRemover, enlever les mots qui ne vehiculent aucun sens
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("text_filtered")

    //troisieme stage: le CountVectorizer
    val vectorizer = new CountVectorizer()
      .setInputCol("text_filtered")
      .setOutputCol("vectorized")

    //quatrieme stage:  output dans une colonne “tfidf”.
    val idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")

    //cinquieme stage:  Conversion de la variable “country2” (catégorielle) en données numériques
    val index_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    //sixieme stage: pareil - conversion de la variable “currency2” en données numériques
    val index_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** VECTOR ASSEMBLER **/

    //septieme stage: Assemblage des variables "tfidf", "days_campaign",
    // "hours_prepa", "goal", "country_indexed", "currency_indexed"  dans une seule colonne “features” grace au VectorAssembler
    val assembler = new VectorAssembler()
        .setInputCols(Array("tfidf", "days_campaign","hours_prepa", "goal", "country_indexed", "currency_indexed"))
        .setOutputCol("features")

    /** MODEL **/
    //huitieme stage: utilisation du type de classifier régression logistique avec les parametres du TP
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/
    //création de la pipeline avec assemblage des précédents stages (variables et modèle classification/regression, dans le bon ordre)
    val pipeline = new Pipeline()
        .setStages(Array(tokenizer,remover,vectorizer,idf,index_country,index_currency,assembler,lr))


    /** TRAINING AND GRID-SEARCH **/

    val Array(training, test) = df.randomSplit(Array[Double](0.9, 0.1)) //on sépare le Data set en deux: 90% (training set) et 10% (test set)

    val param_grid = new ParamGridBuilder() //Gridsearch paramètres minDF et regression_logistique
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(vectorizer.minDF,  Array[Double](55, 75, 95))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val trainValidationSplit = new TrainValidationSplit() // méthode trainValidationSplit sur 70% du Training Set
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(param_grid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(training) // Application de la méthode sur le Training set

    val df_with_predictions = model.transform(test) // Application du modèle sur le Test set

    println("f1_score = " + evaluator.setMetricName("f1").evaluate(df_with_predictions))

    df_with_predictions.groupBy("final_status", "predictions").count.show()

    model.write.overwrite().save("modele-SPARK_TP4-5")





  }
}
