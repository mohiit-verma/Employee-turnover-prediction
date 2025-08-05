from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

def feature_set_3(df):
    """
    Generate promotion-related features for employees
    Features:
    1. promotion_velocity: Average days between promotions
    2. promotion_rate: Number of promotions over a period of time
    3. time_since_last_promotion: Number of days since last promotion from vantage date
    """
    
    print("Starting feature_set_3 generation...")
    
    # Filter data for relevant events before vantage date
    print("Filtering data for promotion/demotion/transfer events before vantage date...")
    filtered_df = df.filter(
        (F.col("event_cd").isin(["PRO", "DEM", "TRF"])) &
        (F.col("event_eff_dt") < F.col("vantage_date"))
    )
    
    # Create window specification for each person ordered by event date
    person_window = Window.partitionBy("person_composit_id").orderBy("event_eff_dt")
    person_window_desc = Window.partitionBy("person_composit_id").orderBy(F.desc("event_eff_dt"))
    
    print("Calculating promotion-specific metrics...")
    
    # Filter for promotions only
    promotions_df = filtered_df.filter(F.col("event_cd") == "PRO")
    
    # Add row numbers and lag functions for promotion calculations
    promotions_with_metrics = promotions_df.withColumn(
        "row_num", F.row_number().over(person_window)
    ).withColumn(
        "prev_promo_date", F.lag("event_eff_dt", 1).over(person_window)
    ).withColumn(
        "days_between_promotions", 
        F.when(F.col("prev_promo_date").isNotNull(),
               F.datediff(F.col("event_eff_dt"), F.col("prev_promo_date")))
    )
    
    print("Generating individual promotion features...")
    
    # Calculate promotion velocity (average days between promotions)
    promotion_velocity = promotions_with_metrics.filter(
        F.col("days_between_promotions").isNotNull()
    ).groupBy("person_composit_id").agg(
        F.avg("days_between_promotions").alias("promotion_velocity")
    )
    
    print("Calculating promotion rate...")
    
    # Calculate promotion rate (total number of promotions)
    promotion_rate = promotions_df.groupBy("person_composit_id").agg(
        F.count("*").alias("promotion_rate")
    )
    
    print("Calculating time since last promotion...")
    
    # Get the most recent promotion date for each person
    last_promotion = promotions_df.withColumn(
        "row_num", F.row_number().over(person_window_desc)
    ).filter(F.col("row_num") == 1).select(
        "person_composit_id", 
        "event_eff_dt", 
        "vantage_date"
    )
    
    # Calculate days since last promotion
    time_since_last_promotion = last_promotion.withColumn(
        "time_since_last_promotion",
        F.datediff(F.col("vantage_date"), F.col("event_eff_dt"))
    ).select("person_composit_id", "time_since_last_promotion")
    
    print("Getting unique persons from original dataset...")
    
    # Get all unique persons with their vantage dates
    unique_persons = df.select("person_composit_id", "vantage_date").distinct()
    
    print("Joining all promotion features...")
    
    # Join all features together
    result_df = unique_persons.join(
        promotion_velocity, "person_composit_id", "left"
    ).join(
        promotion_rate, "person_composit_id", "left"
    ).join(
        time_since_last_promotion, "person_composit_id", "left"
    )
    
    print("Filling null values and finalizing features...")
    
    # Fill null values with appropriate defaults
    final_df = result_df.fillna({
        "promotion_velocity": 0.0,
        "promotion_rate": 0,
        "time_since_last_promotion": 0
    })
    
    # Handle case where promotion_velocity might be null for single promotions
    final_df = final_df.withColumn(
        "promotion_velocity",
        F.when(
            (F.col("promotion_rate") == 1) & (F.col("promotion_velocity") == 0.0),
            F.lit(None).cast("double")
        ).otherwise(F.col("promotion_velocity"))
    )
    
    print("Feature generation completed successfully!")
    print("Features created:")
    print("- promotion_velocity: Average days between promotions")
    print("- promotion_rate: Number of promotions over time period")
    print("- time_since_last_promotion: Days since last promotion from vantage date")
    
    return final_df

