package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseVector, DenseMatrix}
import org.apache.spark.ml.feature.VectorAssembler
import breeze.linalg._
import org.apache.spark.sql.functions.lit

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

    val delta = 0.01
    lazy val data: DataFrame = LinearRegressionTest._training_data
    lazy val hidden_model: DenseVector[Double] = LinearRegressionTest._hidden_model

    private def validateModel(model: LinearRegressionModel): Unit = {
        model.weights.size should be(hidden_model.size)
        model.weights(0) should be(hidden_model(0) +- delta)
        model.weights(1) should be(hidden_model(1) +- delta)
        model.weights(2) should be(hidden_model(2) +- delta)
        model.bias should be(0.0 +- delta)
    }

    "Estimator" should "should produce functional model" in {
        var LRE = new LinearRegressionEstimator()
            .setMaxIter(200)
            .setStepSize(0.5)
        var model = LRE.fit(data)
        validateModel(model)
    }

    "Model" should "make prediction" in {
        var model: LinearRegressionModel = new LinearRegressionModel(
            Vectors.fromBreeze(hidden_model),
            0.0
        )
        .setFeatureCol("features")
        .setLabelCol("label")   
        .setPredictionCol("prediction")
        
        val val_data: DataFrame = data.limit(3)

        val predicted_data: DataFrame = model.transform(val_data)
        predicted_data.collect.foreach(
            row => row(1).asInstanceOf[Double] should be(row(2).asInstanceOf[Double] +- delta)
        )
    }

    "Estimator" should "work after re-read" in {
        val pipeline = new Pipeline().setStages(Array(
            new LinearRegressionEstimator()
                .setMaxIter(200)
                .setStepSize(0.5)
        ))

        val tmpFolder = Files.createTempDir()
        pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
        val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

        val re_read_model = reRead.fit(data.withColumn("prediction", lit(0.0: Double)))
        val model = re_read_model.stages(0).asInstanceOf[LinearRegressionModel]
        validateModel(model)

    }   

    "Model" should "work after re-read" in {
        val pipeline = new Pipeline().setStages(Array(
            new LinearRegressionEstimator()
                .setMaxIter(200)
                .setStepSize(0.5)
                .setPredictionCol("prediction")
        ))

        // I could not solve one problem
        // pipeline was throwing an exception that "prediction" is unavilable
        // .withColumn("prediction", lit(0.0: Double)) - solved this problem 
        // ----------------------------------------------------------------------
        // Failed:
        // - org.apache.spark.ml.made.LinearRegressionTest:
        //     * Model should work after re-read - java.lang.IllegalArgumentException: prediction does not exist. Available: features, label
        // ----------------------------------------------------------------------
        val model: PipelineModel = pipeline.fit(data.withColumn("prediction", lit(0.0: Double)))

        val tmpFolder = Files.createTempDir()
        model.write.overwrite().save(tmpFolder.getAbsolutePath)
        val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)
        
        validateModel(model.stages(0).asInstanceOf[LinearRegressionModel])
    }
}

object LinearRegressionTest extends WithSpark {

    lazy val _DATA_SIZE = 100000
    lazy val _VECTOR_SIZE = 3

    lazy val _rand_data: DenseMatrix[Double] = DenseMatrix.rand(_DATA_SIZE, _VECTOR_SIZE)
    lazy val _hidden_model: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
    lazy val _target: DenseVector[Double] = _rand_data * _hidden_model
    lazy val _total_data: DenseMatrix[Double] = DenseMatrix.horzcat(_rand_data, _target.asDenseMatrix.t)

    lazy val _generated_data = spark.createDataFrame(
        _total_data(*, ::).iterator.map(x => (x(0), x(1), x(2), x(3))).toSeq,
    ).toDF("x1", "x2", "x3", "label")
    
    lazy val _assembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("features")
    lazy val _training_data: DataFrame = _assembler.transform(_generated_data).select("features", "label")
}
