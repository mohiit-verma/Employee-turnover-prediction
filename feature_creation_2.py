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
        F.when(F.col("event_rsn_cd").isin(["USP", "Unsatisfactory Performance", "PER", "Demote - Performance", 
                                          "Performance", "UNS", "Demote Performance", "Unsatisfactory Performance - USP", 
                                          "301", "USI", "UP", "Performance-Driven", "DUP", "PEF", "IPR", "S07", "PNU"]), 1).otherwise(0)
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
    
    # Calculate last 2 years window
    print("Creating 2-year lookback window...")
    two_years_ago = F.date_sub(F.col("vantage_date"), 730)  # 2 years = 730 days
    
    df_with_lookback = df_with_reasons.withColumn(
        "is_within_2_years",
        F.when(F.col("event_eff_dt") >= two_years_ago, 1).otherwise(0)
    )
    
    # Calculate days since last transfer for each person
    print("Calculating days since last transfer...")
    transfer_events = df_with_lookback.filter(F.col("is_transfer") == 1)
    
    # Get the most recent transfer date for each person
    latest_transfer_per_person = transfer_events.groupBy("person_composit_id").agg(
        F.max("event_eff_dt").alias("last_transfer_date")
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
        F.max(F.when((F.col("is_transfer") == 1) & (F.col("is_assignment") == 1), 1).otherwise(0)).alias("transfer_due_to_assignment")
    )
    
    # Get the original dataframe structure (one record per person)
    print("Creating base dataframe with one record per person...")
    base_df = df.select("person_composit_id", "vantage_date").distinct()
    
    # Join aggregated features back to base dataframe
    print("Joining features back to base dataframe...")
    result_df = base_df.join(person_aggregates, ["person_composit_id", "vantage_date"], "left")
    
    # Join days since last transfer
    print("Adding days since last transfer...")
    result_df = result_df.join(latest_transfer_per_person, ["person_composit_id"], "left").withColumn(
        "days_since_last_transfer",
        F.when(F.col("last_transfer_date").isNotNull(),
               F.datediff(F.col("vantage_date"), F.col("last_transfer_date")))
        .otherwise(F.lit(None))
    ).drop("last_transfer_date")
    
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
    return result_df


def validate_features(base_df, result_df):
    """
    Comprehensive validation function for generated features
    
    Args:
        base_df: Original base dataframe
        result_df: Dataframe with generated features
    
    Returns:
        dict: Validation results
    """
    
    print("\n=== STARTING FEATURE VALIDATION ===")
    
    validation_results = {}
    
    # Validation 1: Record count consistency
    print("Validation 1: Checking record count consistency...")
    base_count = base_df.count()
    result_count = result_df.count()
    validation_results['record_count_match'] = base_count == result_count
    print(f"  Base DF records: {base_count}")
    print(f"  Result DF records: {result_count}")
    print(f"  Record count consistency: {'PASS' if validation_results['record_count_match'] else 'FAIL'}")
    
    # Validation 2: One record per person
    print("\nValidation 2: Checking uniqueness of person records...")
    distinct_persons = result_df.select("person_composit_id").distinct().count()
    validation_results['one_record_per_person'] = distinct_persons == result_count
    print(f"  Unique persons: {distinct_persons}")
    print(f"  Total records: {result_count}")
    print(f"  One record per person: {'PASS' if validation_results['one_record_per_person'] else 'FAIL'}")
    
    # Validation 3: Binary feature ranges
    print("\nValidation 3: Checking binary feature value ranges...")
    binary_features = [
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
    
    validation_results['binary_features_valid'] = True
    for col in binary_features:
        min_val = result_df.agg(F.min(col)).collect()[0][0]
        max_val = result_df.agg(F.max(col)).collect()[0][0]
        is_binary = min_val in [0, None] and max_val in [0, 1, None]
        if not is_binary:
            validation_results['binary_features_valid'] = False
        print(f"  {col}: Min={min_val}, Max={max_val}, Binary={'PASS' if is_binary else 'FAIL'}")
    
    # Validation 4: Transfer count validation
    print("\nValidation 4: Checking transfer count validity...")
    min_transfers = result_df.agg(F.min("total_transfers_in_last2_years")).collect()[0][0]
    max_transfers = result_df.agg(F.max("total_transfers_in_last2_years")).collect()[0][0]
    validation_results['transfers_non_negative'] = min_transfers >= 0
    print(f"  Transfer counts - Min: {min_transfers}, Max: {max_transfers}")
    print(f"  Non-negative transfers: {'PASS' if validation_results['transfers_non_negative'] else 'FAIL'}")
    
    # Validation 5: Days since last transfer
    print("\nValidation 5: Checking days since last transfer...")
    min_days = result_df.agg(F.min("days_since_last_transfer")).collect()[0][0]
    max_days = result_df.agg(F.max("days_since_last_transfer")).collect()[0][0]
    non_null_days_count = result_df.filter(F.col("days_since_last_transfer").isNotNull()).count()
    validation_results['days_valid'] = min_days is None or min_days >= 0
    print(f"  Days since transfer - Min: {min_days}, Max: {max_days}")
    print(f"  Non-null records: {non_null_days_count}")
    print(f"  Valid days range: {'PASS' if validation_results['days_valid'] else 'FAIL'}")
    
    # Validation 6: Logical consistency
    print("\nValidation 6: Checking logical consistency...")
    # If transferred_in_last2_years = 1, then total_transfers_in_last2_years >= 1
    inconsistent_records = result_df.filter(
        (F.col("transferred_in_last2_years") == 1) & (F.col("total_transfers_in_last2_years") == 0)
    ).count()
    validation_results['logical_consistency'] = inconsistent_records == 0
    print(f"  Inconsistent transfer records: {inconsistent_records}")
    print(f"  Logical consistency: {'PASS' if validation_results['logical_consistency'] else 'FAIL'}")
    
    # Validation 7: Feature distribution summary
    print("\nValidation 7: Feature distribution summary...")
    promotion_features = result_df.filter(
        (F.col("promotion_due_to_outstanding_performance") == 1) |
        (F.col("promotion_only_title_change") == 1) |
        (F.col("promotion_due_to_market_adjustment") == 1)
    ).count()
    
    demotion_features = result_df.filter(
        (F.col("demotion_due_to_company_reorg_in_last2_years") == 1) |
        (F.col("demotion_due_to_performance_issues_in_last2_years") == 1)
    ).count()
    
    transfer_features = result_df.filter(F.col("transferred_in_last2_years") == 1).count()
    
    print(f"  Records with promotion features: {promotion_features}")
    print(f"  Records with demotion features: {demotion_features}")
    print(f"  Records with transfer features: {transfer_features}")
    
    # Overall validation result
    all_validations_pass = all(validation_results.values())
    validation_results['overall_validation'] = all_validations_pass
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Overall validation result: {'PASS' if all_validations_pass else 'FAIL'}")
    print("=== VALIDATION COMPLETED ===\n")
    
    return validation_results


def print_feature_summary(result_df):
    """
    Print summary statistics for all generated features
    
    Args:
        result_df: DataFrame with generated features
    """
    
    print("\n=== FEATURE SUMMARY ===")
    
    # Binary features summary
    binary_features = [
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
    
    print("Binary Features (Count of 1s):")
    for col in binary_features:
        count_ones = result_df.agg(F.sum(col)).collect()[0][0]
        percentage = (count_ones / result_df.count()) * 100
        print(f"  {col}: {count_ones} ({percentage:.1f}%)")
    
    # Numeric features summary
    print("\nNumeric Features:")
    avg_transfers = result_df.agg(F.avg("total_transfers_in_last2_years")).collect()[0][0]
    max_transfers = result_df.agg(F.max("total_transfers_in_last2_years")).collect()[0][0]
    print(f"  total_transfers_in_last2_years: Average = {avg_transfers:.2f}, Max = {max_transfers}")
    
    # Days since transfer summary
    non_null_days = result_df.filter(F.col("days_since_last_transfer").isNotNull()).count()
    if non_null_days > 0:
        avg_days = result_df.agg(F.avg("days_since_last_transfer")).collect()[0][0]
        print(f"  days_since_last_transfer: Non-null records = {non_null_days}, Average = {avg_days:.1f} days")
    else:
        print(f"  days_since_last_transfer: No records with transfer history")
    
    print("=== FEATURE SUMMARY COMPLETED ===\n")


def main(df):
    """
    Main function to execute the feature generation process
    """
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("EmployeeFeatureGeneration") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Load the data
    print("Loading employee dataset...")
    df = spark.table("table_name")  # Replace with actual table name
    
    print(f"Original dataset loaded with {df.count()} rows")
    
    # Create base dataframe for validation
    base_df = df.select("person_composit_id", "vantage_date").distinct()
    print(f"Unique person-vantage combinations: {base_df.count()}")
    
    # Generate features
    result_df = add_promotion_demotion_transfer_features(df)
    
    print(f"Final dataset with features: {result_df.count()} rows")
    
    # Validate features
    validation_results = validate_features(base_df, result_df)
    
    # Print feature summary
    print_feature_summary(result_df)
    
    # Cache the result for performance if needed for further operations
    result_df.cache()
    
    print("Feature generation and validation process completed successfully!")
    
    return result_df, validation_results
