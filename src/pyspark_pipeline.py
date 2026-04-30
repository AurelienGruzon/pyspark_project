import argparse
import io

import numpy as np
import pandas as pd
from PIL import Image

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, element_at, split, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

from pyspark.ml.functions import array_to_vector
from pyspark.ml.feature import PCA

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model


def build_spark(app_name: str, local: bool):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.parquet.writeLegacyFormat", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "8")
    )

    if local:
        builder = builder.master("local[*]")

    return builder.getOrCreate()


def load_feature_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3),
    )

    for layer in base_model.layers:
        layer.trainable = False

    return Model(
        inputs=base_model.input,
        outputs=base_model.layers[-2].output,
    )


def preprocess_image(content):
    img = Image.open(io.BytesIO(content)).resize((224, 224)).convert("RGB")
    arr = img_to_array(img)
    return preprocess_input(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path: local path or s3://...")
    parser.add_argument("--features-output", required=True, help="Output path for raw features")
    parser.add_argument("--pca-output", required=True, help="Output path for PCA result")
    parser.add_argument("--pca-k", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--local", action="store_true", help="Use local Spark mode")
    args = parser.parse_args()

    spark = build_spark("fruits-pyspark-pipeline", args.local)
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    model = load_feature_model()
    broadcast_weights = sc.broadcast(model.get_weights())

    def model_fn():
        model = load_feature_model()
        model.set_weights(broadcast_weights.value)
        return model

    def featurize_series(model, content_series):
        input_batch = np.stack(content_series.map(preprocess_image))
        preds = model.predict(input_batch, verbose=0)
        return pd.Series([p.flatten().astype(float).tolist() for p in preds])

    @pandas_udf(ArrayType(FloatType()))
    def featurize_udf(content_series: pd.Series) -> pd.Series:
        model = model_fn()
        return featurize_series(model, content_series)

    images = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(args.input)
    )

    images = images.withColumn("label", element_at(split(col("path"), "/"), -2))

    features_df = (
        images
        .repartition(20)
        .select(
            col("path"),
            col("label"),
            featurize_udf(col("content")).alias("features"),
        )
    )

    features_df.write.mode("overwrite").parquet(args.features_output)

    vector_df = features_df.withColumn("features_vector", array_to_vector(col("features")))

    pca = PCA(
        k=args.pca_k,
        inputCol="features_vector",
        outputCol="pca_features",
    )

    pca_model = pca.fit(vector_df)
    pca_df = pca_model.transform(vector_df).select("path", "label", "pca_features")

    pca_df.write.mode("overwrite").parquet(args.pca_output)

    print("Pipeline terminé avec succès.")
    spark.stop()


if __name__ == "__main__":
    main()
