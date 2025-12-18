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
    .appName("LoanDefaultClassification_EDA") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "1g") \
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
df_part1 = spark.read.csv(part1_path, header=True, inferSchema=True,
                          multiLine=True, encoding="UTF-8")
df_part2 = spark.read.csv(part2_path, header=True, inferSchema=True,
                          multiLine=True, encoding="UTF-8")
df_part3 = spark.read.csv(part3_path, header=True, inferSchema=True,
                          multiLine=True, encoding="UTF-8")
df = df_part1.union(df_part2).union(df_part3)

print(f"\nUnified DataFrame dimensions: {df.count():,} x {len(df.columns)}")
print(f"Partitions: {df.rdd.getNumPartitions()}")
df.show(2)
df.printSchema()

# ===============================================
# STEP 4.1: DATA PARTITIONING & SPARK SQL PROCESSING
# ===============================================

print("\n=== STEP 4.1: DATA PARTITIONING & SPARK SQL PROCESSING ===")
df_repartitioned = df.repartition(6)
print(f"Repartitioned to {df_repartitioned.rdd.getNumPartitions()} partitions")
df_repartitioned.createOrReplaceTempView("loans")

spark.sql("""
SELECT
    loan_status,
    COUNT(*) as count,
    AVG(loan_amnt) as mean,
    MIN(loan_amnt) as min,
    MAX(loan_amnt) as max,
    approx_percentile(loan_amnt, 0.25) as q25,
    approx_percentile(loan_amnt, 0.50) as median,
    approx_percentile(loan_amnt, 0.75) as q75,
    STDDEV(loan_amnt) as std
FROM loans
WHERE loan_status IS NOT NULL
GROUP BY loan_status
ORDER BY count DESC
""").show()

df_clean = spark.sql("""
SELECT *
FROM loans
WHERE loan_status IS NOT NULL
""")
print(f"Cleaned rows: {df_clean.count()}")

df = df_clean.withColumn(
    "label",
    when(col("loan_status").isin("Fully Paid"), 0).otherwise(1)
)

df = df.replace(["NONE", "ANY", "OTHER"], "OTHER", subset=["home_ownership"])
crosstab_df = df.crosstab("home_ownership", "label")
crosstab_df.show()
crosstab_df = df.crosstab("grade", "label")
crosstab_df.show()
crosstab_df = df.crosstab("sub_grade", "label")
crosstab_df.show()

df = df.withColumn("zipcode", regexp_extract(col("address"), r"(\d{5})$", 1))
crosstab_df = df.crosstab("zipcode", "label")
crosstab_df.show()
crosstab_df = df.crosstab("verification_status", "label")
crosstab_df.show()
crosstab_df = df.crosstab("term", "label")
crosstab_df.show()
crosstab_df = df.crosstab("purpose", "label")
crosstab_df.show()
crosstab_df = df.crosstab("initial_list_status", "label")
crosstab_df.show()
crosstab_df = df.crosstab("application_type", "label")
crosstab_df.show()

print("\n=== RESOURCE MONITORING ===")
print("Check Spark UI at:", spark.sparkContext.uiWebUrl)
print("Monitor executors, tasks, memory, and shuffle data across nodes")

print("\n=== JOB COMPLETE ===")
spark.stop()
