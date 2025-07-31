from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EmployeeFeatureEngineering") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

def create_employee_features():
    """
    Complete pipeline to create employee promotion/demotion features from raw employee data
    
    Expected table schema:
    - person_composite_id: employee identifier
    - event_cd: event code ('PRO' for promotion, 'DEM' for demotion)
    - vantage_date: reference date
    - event_eff_date: date when promotion/demotion occurred
    - lst_promo_dt: date of last promotion
    """
    
    # Load data from Spark table/catalog
    print("Loading employee data from Spark table...")
    df = spark.table("employee_data")  # Replace with your actual table name
    
    # Show basic info about the dataset
    print(f"Total records loaded: {df.count()}")
    print("Schema:")
    df.printSchema()
    
    print("\nSample data:")
    df.show(5, truncate=False)
    
    # Data quality checks
    print("\nData Quality Checks:")
    print(f"Total records: {df.count()}")
    print(f"Unique employees: {df.select('person_composite_id').distinct().count()}")
    print(f"Records with PRO events: {df.filter(col('event_cd') == 'PRO').count()}")
    print(f"Records with DEM events: {df.filter(col('event_cd') == 'DEM').count()}")
    print(f"Records with null vantage_date: {df.filter(col('vantage_date').isNull()).count()}")
    print(f"Records with null event_eff_date: {df.filter(col('event_eff_date').isNull()).count()}")
    
    # Convert date columns to proper date format if they're strings
    print("\nProcessing date columns...")
    df = df.withColumn("vantage_date", to_date(col("vantage_date"))) \
           .withColumn("event_eff_date", to_date(col("event_eff_date"))) \
           .withColumn("lst_promo_dt", to_date(col("lst_promo_dt")))
    
    # Filter for events that happened before or on vantage date
    print("Filtering events that occurred before vantage date...")
    df_filtered = df.filter(col("event_eff_date") <= col("vantage_date"))
    
    print(f"Records after filtering: {df_filtered.count()}")
    
    # Cache the filtered dataframe as it will be used multiple times
    df_filtered.cache()
    
    # Define window specification for ranking and lag operations
    window_spec = Window.partitionBy("person_composite_id").orderBy("event_eff_date")
    
    print("Creating base aggregated features...")
    
    # Create base features for each employee at each vantage date
    base_features = df_filtered.groupBy("person_composite_id", "vantage_date", "lst_promo_dt").agg(
        # Binary flags for promotions in last 1, 2, 3 years
        max(when((col("event_cd") == "PRO") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -12)), 1)
            .otherwise(0)).alias("promoted_in_last_1y"),
        
        max(when((col("event_cd") == "PRO") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -24)), 1)
            .otherwise(0)).alias("promoted_in_last_2y"),
        
        max(when((col("event_cd") == "PRO") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -36)), 1)
            .otherwise(0)).alias("promoted_in_last_3y"),
        
        # Binary flags for demotions in last 1, 2, 3 years
        max(when((col("event_cd") == "DEM") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -12)), 1)
            .otherwise(0)).alias("demoted_in_last_1y"),
        
        max(when((col("event_cd") == "DEM") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -24)), 1)
            .otherwise(0)).alias("demoted_in_last_2y"),
        
        max(when((col("event_cd") == "DEM") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -36)), 1)
            .otherwise(0)).alias("demoted_in_last_3y"),
        
        # Count of promotions in last 1, 2, 3 years (prefixed with # as per requirement)
        sum(when((col("event_cd") == "PRO") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -12)), 1)
            .otherwise(0)).alias("#_promotions_in_last_1y"),
        
        sum(when((col("event_cd") == "PRO") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -24)), 1)
            .otherwise(0)).alias("#_promotions_in_last_2y"),
        
        sum(when((col("event_cd") == "PRO") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -36)), 1)
            .otherwise(0)).alias("#_promotions_in_last_3y"),
        
        # Count of demotions in last 1, 2, 3 years (prefixed with # as per requirement)
        sum(when((col("event_cd") == "DEM") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -12)), 1)
            .otherwise(0)).alias("#_demotions_in_last_1y"),
        
        sum(when((col("event_cd") == "DEM") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -24)), 1)
            .otherwise(0)).alias("#_demotions_in_last_2y"),
        
        sum(when((col("event_cd") == "DEM") & 
                 (col("event_eff_date") >= add_months(col("vantage_date"), -36)), 1)
            .otherwise(0)).alias("#_demotions_in_last_3y")
    )
    
    print("Adding days since last promotion...")
    
    # Calculate days since last promotion
    base_features = base_features.withColumn(
        "#_of_days_since_last_promotion",
        when((col("lst_promo_dt").isNotNull()) & (col("lst_promo_dt") <= col("vantage_date")),
             datediff(col("vantage_date"), col("lst_promo_dt")))
        .otherwise(None)
    )
    
    print("Calculating average days between promotions...")
    
    # Calculate average days between promotions
    # First, get all promotion dates for each employee
    promotions_only = df_filtered.filter(col("event_cd") == "PRO") \
                                 .select("person_composite_id", "event_eff_date", "vantage_date") \
                                 .distinct()
    
    # Add lag to get previous promotion date
    promotions_with_lag = promotions_only.withColumn(
        "prev_promo_date",
        lag("event_eff_date").over(window_spec)
    )
    
    # Calculate days between consecutive promotions
    promotions_with_days = promotions_with_lag.withColumn(
        "days_between_promos",
        when(col("prev_promo_date").isNotNull(),
             datediff(col("event_eff_date"), col("prev_promo_date")))
    )
    
    # Calculate average days between promotions for each employee
    avg_days_between = promotions_with_days.groupBy("person_composite_id", "vantage_date").agg(
        avg("days_between_promos").alias("avg_days_between_promotion_temp"),
        count("days_between_promos").alias("promotion_count_for_avg")
    )
    
    # Set average to 0 for employees with <= 1 promotion as per requirement
    avg_days_between = avg_days_between.withColumn(
        "avg_days_between_promotion",
        when(col("promotion_count_for_avg") <= 1, 0)
        .otherwise(col("avg_days_between_promotion_temp"))
    ).select("person_composite_id", "vantage_date", "avg_days_between_promotion")
    
    print("Joining all features together...")
    
    # Join all features together
    final_features = base_features.join(
        avg_days_between,
        on=["person_composite_id", "vantage_date"],
        how="left"
    )
    
    # Fill null values for avg_days_between_promotion with 0
    final_features = final_features.fillna({"avg_days_between_promotion": 0})
    
    # Reorder columns to match the original feature specification
    final_features = final_features.select(
        "person_composite_id",
        "vantage_date",
        "promoted_in_last_1y",
        "promoted_in_last_2y", 
        "promoted_in_last_3y",
        "demoted_in_last_1y",
        "demoted_in_last_2y",
        "demoted_in_last_3y",
        "#_promotions_in_last_1y",
        "#_promotions_in_last_2y",
        "#_promotions_in_last_3y",
        "#_demotions_in_last_1y",
        "#_demotions_in_last_2y",
        "#_demotions_in_last_3y",
        "#_of_days_since_last_promotion",
        "avg_days_between_promotion"
    )
    
    # Cache final results
    final_features.cache()
    
    print("Feature engineering completed!")
    print(f"Final feature dataset contains {final_features.count()} records")
    print(f"Number of unique employees in final dataset: {final_features.select('person_composite_id').distinct().count()}")
    
    # Show feature summary statistics
    print("\nFeature Summary Statistics:")
    final_features.describe().show()
    
    # Show sample of final features
    print("\nSample of engineered features:")
    final_features.show(10, truncate=False)
    
    # Unpersist cached dataframes to free memory
    df_filtered.unpersist()
    
    return final_features

def save_features_to_table(features_df, output_table_name):
    """
    Save the engineered features to a Spark table
    
    Parameters:
    features_df: DataFrame with engineered features
    output_table_name: Name of the output table
    """
    
    print(f"Saving features to table: {output_table_name}")
    
    # Write to table (overwrite mode)
    features_df.write \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(output_table_name)
    
    print(f"Features successfully saved to table: {output_table_name}")
    
    # Verify the save operation
    saved_df = spark.table(output_table_name)
    print(f"Verification - Records in saved table: {saved_df.count()}")

def validate_features(features_df):
    """
    Validate the engineered features for data quality
    
    Parameters:
    features_df: DataFrame with engineered features
    """
    
    print("Performing feature validation...")
    
    # Check for negative values in count features
    count_features = ["#_promotions_in_last_1y", "#_promotions_in_last_2y", "#_promotions_in_last_3y",
                     "#_demotions_in_last_1y", "#_demotions_in_last_2y", "#_demotions_in_last_3y"]
    
    for feature in count_features:
        negative_count = features_df.filter(col(feature) < 0).count()
        if negative_count > 0:
            print(f"WARNING: {negative_count} records have negative values for {feature}")
        else:
            print(f"✓ {feature}: No negative values found")
    
    # Check binary features (should only be 0 or 1)
    binary_features = ["promoted_in_last_1y", "promoted_in_last_2y", "promoted_in_last_3y",
                      "demoted_in_last_1y", "demoted_in_last_2y", "demoted_in_last_3y"]
    
    for feature in binary_features:
        invalid_count = features_df.filter(~col(feature).isin([0, 1])).count()
        if invalid_count > 0:
            print(f"WARNING: {invalid_count} records have invalid values for binary feature {feature}")
        else:
            print(f"✓ {feature}: All values are 0 or 1")
    
    # Check logical consistency (3y >= 2y >= 1y for counts)
    consistency_check = features_df.filter(
        (col("#_promotions_in_last_3y") < col("#_promotions_in_last_2y")) |
        (col("#_promotions_in_last_2y") < col("#_promotions_in_last_1y")) |
        (col("#_demotions_in_last_3y") < col("#_demotions_in_last_2y")) |
        (col("#_demotions_in_last_2y") < col("#_demotions_in_last_1y"))
    ).count()
    
    if consistency_check > 0:
        print(f"WARNING: {consistency_check} records have inconsistent time-based counts")
    else:
        print("✓ Time-based count features are logically consistent")
    
    print("Feature validation completed!")

# Main execution
def main():
    """
    Main execution function
    """
    
    try:
        # Create features
        print("Starting employee feature engineering pipeline...")
        features_df = create_employee_features()
        
        # Validate features
        validate_features(features_df)
        
        # Save to table
        output_table_name = "employee_features_engineered"  # Change this to your desired table name
        save_features_to_table(features_df, output_table_name)
        
        # Show final summary
        print("\n" + "="*50)
        print("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Output table: {output_table_name}")
        print(f"Total records: {features_df.count()}")
        print(f"Features created: {len(features_df.columns) - 2}")  # Excluding person_id and vantage_date
        
        # Unpersist final dataframe
        features_df.unpersist()
        
    except Exception as e:
        print(f"Error in feature engineering pipeline: {str(e)}")
        raise e
    
    finally:
        # Stop Spark session
        print("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main()
