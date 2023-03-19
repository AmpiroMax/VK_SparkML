package org.apache.spark.ml.made

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.param.shared.HasStepSize
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.sql.types.DoubleType
import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.lit

trait LinearRegressionParams 
    extends HasFeaturesCol 
    with HasLabelCol
    with HasPredictionCol
    with HasStepSize
    with HasMaxIter
{
    def setLabelCol(value: String): this.type = set(labelCol, value)
    def setFeatureCol(value: String) : this.type = set(featuresCol, value)
    def setPredictionCol(value: String): this.type = set(predictionCol, value)

    setDefault(
        labelCol -> "label",
        featuresCol -> "features",
        predictionCol -> "prediction",
        maxIter -> 500,
        stepSize -> 0.01
    )

    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)
        SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())    

        if (schema.fieldNames.contains($(predictionCol))) {
            SchemaUtils.checkColumnType(schema, getPredictionCol, DoubleType)
            schema
        } else {
            SchemaUtils.appendColumn(schema, schema(getPredictionCol).copy(name = getPredictionCol))
        }
    }
}

class LinearRegressionEstimator(
    override val uid: String
)
    extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable
{
    def setMaxIter(value: Int): this.type = set(maxIter, value)
    def setStepSize(value: Double): this.type = set(stepSize, value)
 
    def this() = this(Identifiable.randomUID("LinearRegressionEstimator"))

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

    override def fit(dataset: Dataset[_]): LinearRegressionModel = {
        implicit val encoder : Encoder[Vector] = ExpressionEncoder()

        val data_with_bias: Dataset[_] = dataset.withColumn("bias_feature", lit(1))
        val assembler: VectorAssembler = new VectorAssembler()
            .setInputCols(Array("bias_feature", $(featuresCol), $(labelCol)))
            .setOutputCol("features_with_bias")
        val data: Dataset[Vector] = assembler.transform(data_with_bias).select("features_with_bias").as[Vector]
        
        val feature_size: Int = data.first().size - 1
        var weights: BreezeDenseVector[Double] = BreezeDenseVector.rand[Double](feature_size)

        for (_ <- 0 to getMaxIter) {
            val gradient = data.rdd.mapPartitions((data_part: Iterator[Vector]) => {
                val partial_gradient = new MultivariateOnlineSummarizer()
                data_part.foreach(vector => {
                    val x = vector.asBreeze(0 until weights.size).toDenseVector
                    val y = vector.asBreeze(weights.size)
                    partial_gradient.add(fromBreeze(x *:* (sum(x *:* weights) - y)))
                })
                Iterator(partial_gradient)
            }).reduce(_ merge _)
            weights = weights - $(stepSize) * gradient.mean.asBreeze
        }

        copyValues(
            new LinearRegressionModel(Vectors.fromBreeze(weights(1 until feature_size)), weights(0))
        ).setParent(this)
    }

}

object LinearRegressionEstimator extends DefaultParamsReadable[LinearRegressionEstimator]

class LinearRegressionModel private[made](
    override val uid: String,
    val weights: DenseVector,
    val bias: Double
)
    extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable
{
    private[made] def this(weights: Vector, bias: Double) =
        this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
        new LinearRegressionModel(weights, bias), extra
    )

    override def transform(dataset: Dataset[_]): DataFrame = {
        val Bweights = weights.asBreeze
        val bStds = bias
        val transformUdf = {
            dataset.sqlContext.udf.register(
                uid + "_transform",
                (x : Vector) => {
                    Bweights.dot(x.asBreeze) + bias
                }
            )
        }
        dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
    }
    
    override def write: MLWriter = new DefaultParamsWriter(this) {
        override protected def saveImpl(path: String): Unit = {
            super.saveImpl(path)
            val vectors: (Vector, Double) = weights.asInstanceOf[Vector] -> bias.asInstanceOf[Double]
            sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
        }
    }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
    override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
        override def load(path: String): LinearRegressionModel = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc)
            val vectors = sqlContext.read.parquet(path + "/vectors")
            implicit val encoder_vec: Encoder[Vector] = ExpressionEncoder()
            implicit val encoder_doub: Encoder[Double] = ExpressionEncoder()

            val (weights, bias) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

            val model = new LinearRegressionModel(weights, bias)
            metadata.getAndSetParams(model)
            model
        }
    }
}