#!/usr/bin/env python3

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_extract
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import socket

# =========================
# STEP 1: CLUSTER CONFIG
# =========================

spark = SparkSession.builder \
    .appName("LoanDefaultClassification_LR") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512mb") \
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

# ======================================================
# STEPS 2–3: MANUAL DATA SPLITTING + DISTRIBUTED LOADING
# ======================================================

print("\n=== STEPS 2-3: MANUAL DATA SPLITTING + DISTRIBUTED LOADING ===")
part1_path = "/opt/spark/work-dir/data/archive/lending_club_part_1.csv"
part2_path = "/opt/spark/work-dir/data/archive/lending_club_part_2.csv"
part3_path = "/opt/spark/work-dir/data/archive/lending_club_part_3.csv"


print("Loading ALL dataset parts across cluster:")
df_part1 = spark.read.csv(part1_path, header=True, inferSchema=True, multiLine=True, encoding="UTF-8")
df_part2 = spark.read.csv(part2_path, header=True, inferSchema=True, multiLine=True, encoding="UTF-8")
df_part3 = spark.read.csv(part3_path, header=True, inferSchema=True, multiLine=True, encoding="UTF-8")
df = df_part1.union(df_part2).union(df_part3)

#print("Loading Unified dataset parts across cluster:")
#datapath = "/opt/spark/work-dir/data/archive/lending_club_loan_two.csv"
#df = spark.read.csv(datapath, header=True, inferSchema=True, multiLine=True, encoding="UTF-8")

print(f"\nUnified DataFrame dimensions: {df.count():,} x {len(df.columns)}")
print(f"Partitions: {df.rdd.getNumPartitions()}")
df.show(2)
df.printSchema()



df = df.filter(df.loan_status.isNotNull())

df = df.withColumn(
    "label",
    when(col("loan_status").isin("Fully Paid"), 0).otherwise(1)
)

df = df.replace(["NONE", "ANY", "OTHER"], "OTHER", subset=["home_ownership"])

df = df.withColumn("zipcode", regexp_extract(col("address"), r"(\d{5})$", 1))

# ==========================
# STEP 4.2: FEATURE ENGINEERING
# ==========================

print("\n=== STEP 4.2. FEATURE ENGINEERING ===")
categorical_cols = ["zipcode"]

numeric_cols = [
    "installment", "annual_inc", "dti"
    ,    "open_acc", "pub_rec", "revol_bal", "total_acc"
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
feature_assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)

pipeline = Pipeline(stages=indexers + encoders + [feature_assembler, lr])

# ==========================
# STEP 6: TRAIN–TEST SPLIT (NO SEED FOR VARIABILITY)
# ==========================

print("\n=== 6. DISTRIBUTED TRAINING and TUNING THE ML MODEL (CV + HYPERPARAMETERS) ===")
train_df, test_df = df.randomSplit([0.8, 0.2])  # REMOVED seed=42 for variability

# ================================
# STEP 8: TUNING WITH CROSS-VALIDATION (EXPANDED GRID)
# ===============================

bin_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

paramGrid = (ParamGridBuilder()
    .addGrid(lr.maxIter, [10, 20, 30])  
    .addGrid(lr.regParam, [0.001, 0.01, 0.1])  
    .addGrid(lr.elasticNetParam, [0.0, 0.3, 0.8])  
    .build()
)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=bin_evaluator,
    numFolds=2,  
    parallelism=2  
)

cv_start = time.time()
sample_df = train_df.sample(withReplacement=False, fraction=0.2)  # seed=42
cv_model = cv.fit(sample_df)
cv_end = time.time()

print(f"CV Training Time: {cv_end - cv_start:.3f} seconds")

best_model = cv_model.bestModel
best_lr = best_model.stages[-1]
print("\nBest Hyperparameters from CV:")
print(" maxIter :", best_lr.getMaxIter())
print(" regParam :", best_lr.getRegParam())
print(" elasticNetParam:", best_lr.getElasticNetParam())

# ==========================
# STEP 7: PREDICTIONS
# ==========================

print("\n=== 7. PREDICTIONS WITH BEST MODEL ===")
print("Training set count:", train_df.count())
pred_start = time.time()
train_predictions = best_model.transform(train_df)
pred_end = time.time()
print(f"Training Time (Prediction): {pred_end - pred_start:.3f} seconds")

print("Test set count:", test_df.count())
pred_start = time.time()
test_predictions = best_model.transform(test_df)
pred_end = time.time()
print(f"Testing Time (Prediction): {pred_end - pred_start:.3f} seconds")

# ==========================
# STEP 8 (EVAL): METRICS
# ==========================

print("\n=== 8. MODEL EVALUATION (TUNED MODEL) ===")
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

train_accuracy = evaluator.setMetricName("accuracy").evaluate(train_predictions)
train_f1 = evaluator.setMetricName("f1").evaluate(train_predictions)
train_weighted_precision = evaluator.setMetricName("weightedPrecision").evaluate(train_predictions)
train_weighted_recall = evaluator.setMetricName("weightedRecall").evaluate(train_predictions)

test_accuracy = evaluator.setMetricName("accuracy").evaluate(test_predictions)
test_f1 = evaluator.setMetricName("f1").evaluate(test_predictions)
test_weighted_precision = evaluator.setMetricName("weightedPrecision").evaluate(test_predictions)
test_weighted_recall = evaluator.setMetricName("weightedRecall").evaluate(test_predictions)

print("Metric".ljust(20), "Train Set".ljust(15), "Test Set".ljust(15))
print("-" * 50)
print("Accuracy".ljust(20), f"{train_accuracy:.2f}".ljust(15), f"{test_accuracy:.2f}".ljust(15))
print("F1 Score".ljust(20), f"{train_f1:.2f}".ljust(15), f"{test_f1:.2f}".ljust(15))
print("Weighted Precision".ljust(20), f"{train_weighted_precision:.2f}".ljust(15), f"{test_weighted_precision:.2f}".ljust(15))
print("Weighted Recall".ljust(20), f"{train_weighted_recall:.2f}".ljust(15), f"{test_weighted_recall:.2f}".ljust(15))

# ==========================
# RESOURCE MONITORING
# ==========================

print("\n=== RESOURCE MONITORING ===")
print("Check Spark UI at:", spark.sparkContext.uiWebUrl)
print("Monitor executors, tasks, memory, and shuffle data across nodes")

print("\n=== JOB COMPLETE ===")
spark.stop()
