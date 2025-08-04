from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

def add_promotion_demotion_transfer_features(df):
    """
    Generate promotion, demotion, and transfer features for employee dataset
    
    Args:
        df: PySpark DataFrame containing employee data
    
    Returns:
        PySpark DataFrame with additional features
    """
    
    print("Starting feature generation for promotion, demotion, and transfer...")
    
    # Filter data to include only events before vantage_date
    print("Filtering events that occurred before vantage_date...")
    df_filtered = df.filter(F.col("event_eff_dt") < F.col("vantage_date"))
    
    # Create binary flags for event types
    print("Creating event type flags...")
    df_with_flags = df_filtered.withColumn(
        "is_promotion", 
        F.when(F.col("event_cd") == "PRO", 1).otherwise(0)
    ).withColumn(
        "is_demotion", 
        F.when(F.col("event_cd") == "DEM", 1).otherwise(0)
    ).withColumn(
        "is_transfer", 
        F.when(F.col("event_cd") == "XFR", 1).otherwise(0)
    )
    
    # Create reason code flags for detailed categorization
    print("Creating detailed reason code flags...")
    df_with_reasons = df_with_flags.withColumn(
        "is_outstanding_performance",
        F.when(F.col("event_rsn_cd").isin(["OPR", "Outstanding Performance", "PTP", "OP", "PER"]), 1).otherwise(0)
    ).withColumn(
        "is_title_change_only",
        F.when(F.col("event_rsn_cd").isin(["JOB", "TC", "T", "PRO", "M28", "TCH", "TTL", "PTC", "PNP", "JTC", "OFC"]), 1).otherwise(0)
    ).withColumn(
        "is_market_adjustment",
        F.when(F.col("event_rsn_cd").isin(["MKT", "MRK", "MKA"]), 1).otherwise(0)
    ).withColumn(
        "is_company_reorg",
        F.when(F.col("event_rsn_cd").isin(["RES", "REO", "MOR", "TRP", "ORG", "MPC", "ROS"]), 1).otherwise(0)
    ).withColumn(
        "is_performance_issues",
        F.when(F.col("event_rsn_cd").isin(["USP", "Unsatisfactory Performance", "PER", "Demote - Performance", "Performance", "UNS", "Demote Performance", "Unsatisfactory Performance - USP", "301", "USI", "UP", "Performance-Driven", "DUP", "PEF", "IPR", "307", "PNU"]), 1).otherwise(0)
    ).withColumn(
        "is_employee_request",
        F.when(F.col("event_rsn_cd").isin(["EER", "Employee Request", "ER2", "ER1", "EE"]), 1).otherwise(0)
    ).withColumn(
        "is_skill_based",
        F.when(F.col("event_rsn_cd").contains("Transfer - Skill-based"), 1).otherwise(0)
    ).withColumn(
        "is_relocation",
        F.when(F.col("event_rsn_cd").isin(["Relocation", "REL"]), 1).otherwise(0)
    ).withColumn(
        "is_assignment",
        F.when(F.col("event_rsn_cd").isin(["TMP", "Expatriate Assignment", "ASC", "EXP", "1", "SAB", "SPA", "IPA"]), 1).otherwise(0)
    )
    
    # Define window for time-based calculations
    window_person = Window.partitionBy("person_composit_id").orderBy("event_eff_dt")
    window_person_desc = Window.partitionBy("person_composit_id").orderBy(F.desc("event_eff_dt"))
    
    # Calculate days since last transfer
    print("Calculating days since last transfer...")
    # First, create a dataset with only transfer events to calculate previous transfer dates
    transfer_events = df_with_reasons.filter(F.col("is_transfer") == 1)
    
    # Add previous transfer date for transfer events only
    transfer_with_prev = transfer_events.withColumn(
        "prev_transfer_date",
        F.lag("event_eff_dt").over(window_person)
    )
    
    # Calculate days since last transfer for each person
    # Get the most recent transfer date for each person
    latest_transfer_per_person = transfer_with_prev.groupBy("person_composit_id").agg(
        F.max("event_eff_dt").alias("last_transfer_date")
    )
    
    # Join back to main dataset and calculate days since last transfer
    df_with_transfer_days = df_with_reasons.join(
        latest_transfer_per_person, 
        ["person_composit_id"], 
        "left"
    ).withColumn(
        "days_since_last_transfer",
        F.when(F.col("last_transfer_date").isNotNull(),
               F.datediff(F.col("vantage_date"), F.col("last_transfer_date")))
        .otherwise(F.lit(None))
    ).drop("last_transfer_date")
    
    # Calculate last 2 years window
    print("Creating 2-year lookback window...")
    two_years_ago = F.date_sub(F.col("vantage_date"), 730)  # 2 years = 730 days
    
    df_with_lookback = df_with_transfer_days.withColumn(
        "is_within_2_years",
        F.when(F.col("event_eff_dt") >= two_years_ago, 1).otherwise(0)
    )
    
    # Generate aggregate features per person
    print("Generating aggregate features per person...")
    person_aggregates = df_with_lookback.groupBy("person_composit_id", "vantage_date").agg(
        # Promotion features
        F.max(F.when((F.col("is_promotion") == 1) & (F.col("is_outstanding_performance") == 1), 1).otherwise(0)).alias("promotion_due_to_outstanding_performance"),
        F.max(F.when((F.col("is_promotion") == 1) & (F.col("is_title_change_only") == 1), 1).otherwise(0)).alias("promotion_only_title_change"),
        F.max(F.when((F.col("is_promotion") == 1) & (F.col("is_market_adjustment") == 1), 1).otherwise(0)).alias("promotion_due_to_market_adjustment"),
        
        # Demotion features
        F.max(F.when((F.col("is_demotion") == 1) & (F.col("is_company_reorg") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("demotion_due_to_company_reorg_in_last2_years"),
        F.max(F.when((F.col("is_demotion") == 1) & (F.col("is_performance_issues") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("demotion_due_to_performance_issues_in_last2_years"),
        
        # Transfer features
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("transferred_in_last2_years"),
        F.sum(F.when((F.col("is_transfer") == 1) & (F.col("is_within_2_years") == 1), 1).otherwise(0)).alias("total_transfers_in_last2_years"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_company_reorg") == 1), 1).otherwise(0)).alias("transfer_due_to_company_reorg"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_employee_request") == 1), 1).otherwise(0)).alias("transfer_due_to_employee_request"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_skill_based") == 1), 1).otherwise(0)).alias("transfer_based_on_skill"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_relocation") == 1), 1).otherwise(0)).alias("transfer_due_to_employee_relocation"),
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_assignment") == 1), 1).otherwise(0)).alias("transfer_due_to_assignment"),
        
        # Days since last transfer
        F.min("days_since_last_transfer").alias("days_since_last_transfer")
    )
    
    # Get the original dataframe structure (one record per person)
    print("Creating base dataframe with one record per person...")
    base_df = df.select("person_composit_id", "vantage_date").distinct()
    
    # Join aggregated features back to base dataframe
    print("Joining features back to base dataframe...")
    result_df = base_df.join(person_aggregates, ["person_composit_id", "vantage_date"], "left")
    
    # Fill null values with 0 for binary features and appropriate defaults for others
    print("Filling null values with appropriate defaults...")
    feature_columns = [
        "promotion_due_to_outstanding_performance",
        "promotion_only_title_change", 
        "promotion_due_to_market_adjustment",
        "demotion_due_to_company_reorg_in_last2_years",
        "demotion_due_to_performance_issues_in_last2_years",
        "transferred_in_last2_years",
        "transfer_due_to_company_reorg",
        "transfer_due_to_employee_request",
        "transfer_based_on_skill",
        "transfer_due_to_employee_relocation",
        "transfer_due_to_assignment"
    ]
    
    for col in feature_columns:
        result_df = result_df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    
    # Fill total_transfers_in_last2_years with 0
    result_df = result_df.withColumn("total_transfers_in_last2_years", F.coalesce(F.col("total_transfers_in_last2_years"), F.lit(0)))
    
    print("Feature generation completed successfully!")
    
    # Validation section
    print("\n=== FEATURE VALIDATION ===")
    
    # Validate record count
    base_count = base_df.count()
    result_count = result_df.count()
    print(f"Validation 1 - Record Count: Base DF: {base_count}, Result DF: {result_count}")
    print(f"Record count consistency: {'PASS' if base_count == result_count else 'FAIL'}")
    
    # Validate no duplicate person records
    distinct_persons = result_df.select("person_composit_id").distinct().count()
    print(f"Validation 2 - Unique persons: {distinct_persons}")
    print(f"One record per person: {'PASS' if distinct_persons == result_count else 'FAIL'}")
    
    # Validate binary features are indeed binary (0 or 1)
    print("Validation 3 - Binary feature ranges:")
    for col in feature_columns:
        min_val = result_df.agg(F.min(col)).collect()[0][0]
        max_val = result_df.agg(F.max(col)).collect()[0][0]
        is_binary = min_val in [0, None] and max_val in [0, 1, None]
        print(f"  {col}: Min={min_val}, Max={max_val}, Binary={'PASS' if is_binary else 'FAIL'}")
    
    # Validate transfer count is non-negative
    min_transfers = result_df.agg(F.min("total_transfers_in_last2_years")).collect()[0][0]
    max_transfers = result_df.agg(F.max("total_transfers_in_last2_years")).collect()[0][0]
    print(f"Validation 4 - Transfer counts: Min={min_transfers}, Max={max_transfers}")
    print(f"Non-negative transfers: {'PASS' if min_transfers >= 0 else 'FAIL'}")
    
    # Validate days since last transfer
    min_days = result_df.agg(F.min("days_since_last_transfer")).collect()[0][0]
    max_days = result_df.agg(F.max("days_since_last_transfer")).collect()[0][0]
    print(f"Validation 5 - Days since transfer: Min={min_days}, Max={max_days}")
    
    # Logical validation: if transferred_in_last2_years = 1, then total_transfers_in_last2_years >= 1
    logical_check = result_df.filter(
        (F.col("transferred_in_last2_years") == 1) & (F.col("total_transfers_in_last2_years") == 0)
    ).count()
    print(f"Validation 6 - Logical consistency (transferred flag vs count): Inconsistent records={logical_check}")
    print(f"Logical consistency: {'PASS' if logical_check == 0 else 'FAIL'}")
    
    print("=== VALIDATION COMPLETED ===\n")
    
    return result_df

def main(df):
    """
    Main function to execute the feature generation process
    """
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("EmployeeFeatureGeneration") \
        .getOrCreate()
    
    print(f"Original dataset shape: {df.count()} rows")
    
    # Generate features
    result_df = add_promotion_demotion_transfer_features(df)
    
    print(f"Final dataset with features: {result_df.count()} rows")
    
    # Show feature summary
    print("\nFeature Summary:")
    feature_cols = [
        "promotion_due_to_outstanding_performance",
        "promotion_only_title_change", 
        "promotion_due_to_market_adjustment",
        "demotion_due_to_company_reorg_in_last2_years",
        "demotion_due_to_performance_issues_in_last2_years",
        "transferred_in_last2_years",
        "total_transfers_in_last2_years",
        "transfer_due_to_company_reorg",
        "transfer_due_to_employee_request",
        "transfer_based_on_skill",
        "transfer_due_to_employee_relocation",
        "transfer_due_to_assignment",
        "days_since_last_transfer"
    ]
    
    for col in feature_cols:
        if col == "total_transfers_in_last2_years":
            avg_val = result_df.agg(F.avg(col)).collect()[0][0]
            print(f"{col}: Average = {avg_val:.2f}")
        elif col == "days_since_last_transfer":
            non_null_count = result_df.filter(F.col(col).isNotNull()).count()
            print(f"{col}: Non-null records = {non_null_count}")
        else:
            sum_val = result_df.agg(F.sum(col)).collect()[0][0]
            print(f"{col}: Total = {sum_val}")
    
    return result_df
