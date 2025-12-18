# scenario2_integrated_two_jobs.py

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_extract
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# --------------------------------------------------
# 1. Spark session
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("Scenario2_LR") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "1gb") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "3") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "6") \
    .config("spark.sql.shuffle.partitions", "6") \
    .config("spark.sql.files.maxPartitionBytes", "16MB") \
    .config("spark.driver.memory", "1g") \
    .config("spark.driver.cores", "1") \
    .config("spark.worker.cleanup.enabled", "true") \
    .getOrCreate()

print("=== STEP 1: CLUSTER CONFIGURATION ===")
print("Master:", spark.sparkContext.master)
print("Web UI:", spark.sparkContext.uiWebUrl)
print("Executor memory:", spark.conf.get("spark.executor.memory", "Not set"))
print("Executor cores:", spark.conf.get("spark.executor.cores", "Not set"))
print("Shuffle partitions:", spark.conf.get("spark.sql.shuffle.partitions", "Not set"))

# --------------------------------------------------
# 2. Load unified (integrated) dataset
#    In your cluster this would be the full lending_club_loan_two.csv
# --------------------------------------------------
print("=== STEP 2: LOADING DATA ===")
datapath = "/opt/spark/work-dir/data/archive/lending_club_loan_two.csv"
df = spark.read.csv(datapath, header=True, inferSchema=True, multiLine=True, encoding="UTF-8")

print(f"\nUnified DataFrame dimensions: {df.count():,} x {len(df.columns)}")
print(f"Partitions: {df.rdd.getNumPartitions()}")
df.show(2)
df.printSchema()

# --------------------------------------------------
# 3. Basic label preparation (example)
#    Convert loan_status to binary label: 1 = Fully Paid, 0 = Charged Off
#    Adapt this mapping to your actual data.
# --------------------------------------------------
print("=== STEP 3: DATA PROCESSING ===")
df = df.filter(df.loan_status.isNotNull())

df = df.withColumn("zipcode", regexp_extract(col("address"), r"(\d{5})$", 1))

df = df.withColumn(
    "label",
    when(col("loan_status") == "Fully Paid", 1.0)
    .when(col("loan_status") == "Charged Off", 0.0)
    .otherwise(None)
)

df = df.na.drop(subset=["label"])

# --------------------------------------------------
# 4. Feature selection & preprocessing (example)
#    Adjust to match what you did in Part One.
# --------------------------------------------------
print("\n=== STEP 4: FEATURE ENGINEERING ===")
categorical_cols = ["zipcode"]

numeric_cols = [
    "installment", "annual_inc", "dti"
]

for c in numeric_cols:
    df = df.withColumn(c, col(c).cast("double"))

# ===============================
# STEP 5: BUILDING ML PIPELINE
# ===============================

print("\n=== 5. BUILDING ML PIPELINE ===")
broadcast_categorical_cols = spark.sparkContext.broadcast(categorical_cols)

indexers = [
    StringIndexer(
        inputCol=col_name,
        outputCol=f"{col_name}_idx",
        handleInvalid="keep"
    )
    for col_name in broadcast_categorical_cols.value
]

encoders = [
    OneHotEncoder(
        inputCol=f"{col_name}_idx",
        outputCol=f"{col_name}_vec"
    )
    for col_name in broadcast_categorical_cols.value
]

assembler_inputs = [f"{col_name}_vec" for col_name in broadcast_categorical_cols.value] + numeric_cols

assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features",
    handleInvalid="keep"
)

# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------
print("\n=== 6. TRAINING ===")
train_df, test_df = df.randomSplit([0.7, 0.3])

# --------------------------------------------------
# 6. Define two pipelines: Logistic Regression & Random Forest
# --------------------------------------------------
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

lr_pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

# --------------------------------------------------
# 7. Train both models "simultaneously"
#    When both fit calls are triggered without waiting on actions from the other,
#    Spark will schedule them as concurrent jobs (depending on cluster config).
# --------------------------------------------------
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

print("Training set count:", train_df.count())
# Start time for LR
start_lr = time.time()
lr_model = lr_pipeline.fit(train_df)
lr_training_time = time.time() - start_lr


# --------------------------------------------------
# 8. Evaluate both models on the same test set
# --------------------------------------------------
print("Testing set count:", test_df.count())
pred_start = time.time()
lr_predictions = lr_model.transform(test_df)
pred_end = time.time()
print(f"LR Testing Time (Prediction): {pred_end - pred_start:.3f} seconds")

lr_accuracy = evaluator.evaluate(lr_predictions)

print("\n=== Scenario 2: Integrated Dataset, Two Jobs ===")
print(f"Logistic Regression - Training time (s): {lr_training_time:.2f}")
print(f"Logistic Regression - Test Accuracy:     {lr_accuracy:.4f}")

# --------------------------------------------------
# 9. Keep the application running a bit (optional)
#    to inspect Spark UI for concurrent jobs if needed
# --------------------------------------------------
# import time; time.sleep(600)

spark.stop()
