from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

def create_employee_features(df):
    """
    Create employee promotion/demotion features from raw employee data
    Returns one record per person_composite_id with all engineered features
    
    Parameters:
    df: Input PySpark DataFrame with columns:
        - person_composite_id: employee identifier
        - event_cd: event code ('PRO' for promotion, 'DEM' for demotion)
        - vantage_date: reference date
        - event_eff_dt: date when promotion/demotion occurred
        - lst_promo_dt: date of last promotion
    
    Returns:
    DataFrame with one record per employee containing all features
    """
    
    # Convert date columns to proper date format if they're strings
    df = df.withColumn("vantage_date", to_date(col("vantage_date"))) \
           .withColumn("event_eff_dt", to_date(col("event_eff_dt"))) \
           .withColumn("lst_promo_dt", to_date(col("lst_promo_dt")))
    
    # Filter for events that happened before or on vantage date
    df_filtered = df.filter(col("event_eff_dt") <= col("vantage_date"))
    
    # Cache the filtered dataframe as it will be used multiple times
    df_filtered.cache()
    
    # Define window specifications
    window_spec_person = Window.partitionBy("person_composite_id")
    window_spec_person_ordered = Window.partitionBy("person_composite_id").orderBy("event_eff_dt")
    
    # Step 1: Generate all features for each record
    features_expanded = df_filtered.withColumn(
        # Binary flags for promotions in last 1, 2, 3 years
        "promoted_in_last_1y",
        when((col("event_cd") == "PRO") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -12)), 1).otherwise(0)
    ).withColumn(
        "promoted_in_last_2y",
        when((col("event_cd") == "PRO") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -24)), 1).otherwise(0)
    ).withColumn(
        "promoted_in_last_3y",
        when((col("event_cd") == "PRO") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -36)), 1).otherwise(0)
    ).withColumn(
        # Binary flags for demotions in last 1, 2, 3 years
        "demoted_in_last_1y",
        when((col("event_cd") == "DEM") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -12)), 1).otherwise(0)
    ).withColumn(
        "demoted_in_last_2y",
        when((col("event_cd") == "DEM") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -24)), 1).otherwise(0)
    ).withColumn(
        "demoted_in_last_3y",
        when((col("event_cd") == "DEM") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -36)), 1).otherwise(0)
    ).withColumn(
        # Count features (initially 1 or 0 for each record)
        "#_promotions_in_last_1y",
        when((col("event_cd") == "PRO") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -12)), 1).otherwise(0)
    ).withColumn(
        "#_promotions_in_last_2y",
        when((col("event_cd") == "PRO") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -24)), 1).otherwise(0)
    ).withColumn(
        "#_promotions_in_last_3y",
        when((col("event_cd") == "PRO") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -36)), 1).otherwise(0)
    ).withColumn(
        "#_demotions_in_last_1y",
        when((col("event_cd") == "DEM") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -12)), 1).otherwise(0)
    ).withColumn(
        "#_demotions_in_last_2y",
        when((col("event_cd") == "DEM") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -24)), 1).otherwise(0)
    ).withColumn(
        "#_demotions_in_last_3y",
        when((col("event_cd") == "DEM") & 
             (col("event_eff_dt") >= add_months(col("vantage_date"), -36)), 1).otherwise(0)
    )
    
    # **LINES 97-105: UPDATED LOGIC FOR DAYS SINCE LAST PROMOTION**
    # First, get the most recent promotion date for each person from event records
    latest_promotion_from_events = features_expanded.filter(col("event_cd") == "PRO") \
        .withColumn("latest_promo_from_events", 
                   min("event_eff_dt").over(window_spec_person)) \
        .select("person_composite_id", "latest_promo_from_events").distinct()
    
    # Join back to get the latest promotion date from events
    features_expanded = features_expanded.join(
        latest_promotion_from_events, on="person_composite_id", how="left"
    ).withColumn(
        # Days since last promotion with updated logic
        "#_of_days_since_last_promotion",
        when(col("lst_promo_dt").isNotNull() & (col("lst_promo_dt") <= col("vantage_date")),
             datediff(col("vantage_date"), col("lst_promo_dt")))
        .when(col("lst_promo_dt").isNull() & col("latest_promo_from_events").isNotNull() & 
              (col("latest_promo_from_events") <= col("vantage_date")),
             datediff(col("vantage_date"), col("latest_promo_from_events")))
        .otherwise(None)
    )
    
    # Step 2: Calculate promotion intervals for average days calculation
    promotion_records = features_expanded.filter(col("event_cd") == "PRO")
    
    # Add previous promotion date using window function
    promotion_with_lag = promotion_records.withColumn(
        "prev_promo_date",
        lag("event_eff_dt").over(window_spec_person_ordered)
    ).withColumn(
        "days_between_current_and_prev",
        when(col("prev_promo_date").isNotNull(),
             datediff(col("event_eff_dt"), col("prev_promo_date"))).otherwise(None)
    )
    
    # Calculate running statistics for average days between promotions
    promotion_with_stats = promotion_with_lag.withColumn(
        "promotion_rank",
        row_number().over(window_spec_person_ordered)
    ).withColumn(
        "total_promotions_for_person",
        count("*").over(window_spec_person)
    ).withColumn(
        "sum_days_between_promos",
        sum("days_between_current_and_prev").over(window_spec_person)
    ).withColumn(
        "count_intervals",
        count("days_between_current_and_prev").over(window_spec_person)
    ).withColumn(
        "avg_days_between_promotion_temp",
        when(col("count_intervals") > 0, 
             col("sum_days_between_promos") / col("count_intervals")).otherwise(0)
    ).withColumn(
        "avg_days_between_promotion",
        when(col("total_promotions_for_person") <= 1, 0)
        .otherwise(col("avg_days_between_promotion_temp"))
    )
    
    # Step 3: Join promotion statistics back to main dataset
    promotion_avg_only = promotion_with_stats.select(
        "person_composite_id", 
        "vantage_date",
        "avg_days_between_promotion"
    ).distinct()
    
    # Join average days back to main features
    features_with_avg = features_expanded.join(
        promotion_avg_only,
        on=["person_composite_id", "vantage_date"],
        how="left"
    ).fillna({"avg_days_between_promotion": 0})
    
    # Step 4: Aggregate all features per person using window functions to get final values
    features_aggregated = features_with_avg.withColumn(
        # Get latest vantage date for each person
        "latest_vantage_date",
        max("vantage_date").over(window_spec_person)
    ).withColumn(
        # Aggregate binary flags (max value per person)
        "promoted_in_last_1y_final",
        max("promoted_in_last_1y").over(window_spec_person)
    ).withColumn(
        "promoted_in_last_2y_final",
        max("promoted_in_last_2y").over(window_spec_person)
    ).withColumn(
        "promoted_in_last_3y_final",
        max("promoted_in_last_3y").over(window_spec_person)
    ).withColumn(
        "demoted_in_last_1y_final",
        max("demoted_in_last_1y").over(window_spec_person)
    ).withColumn(
        "demoted_in_last_2y_final",
        max("demoted_in_last_2y").over(window_spec_person)
    ).withColumn(
        "demoted_in_last_3y_final",
        max("demoted_in_last_3y").over(window_spec_person)
    ).withColumn(
        # Aggregate count features (sum per person)
        "#_promotions_in_last_1y_final",
        sum("#_promotions_in_last_1y").over(window_spec_person)
    ).withColumn(
        "#_promotions_in_last_2y_final",
        sum("#_promotions_in_last_2y").over(window_spec_person)
    ).withColumn(
        "#_promotions_in_last_3y_final",
        sum("#_promotions_in_last_3y").over(window_spec_person)
    ).withColumn(
        "#_demotions_in_last_1y_final",
        sum("#_demotions_in_last_1y").over(window_spec_person)
    ).withColumn(
        "#_demotions_in_last_2y_final",
        sum("#_demotions_in_last_2y").over(window_spec_person)
    ).withColumn(
        "#_demotions_in_last_3y_final",
        sum("#_demotions_in_last_3y").over(window_spec_person)
    ).withColumn(
        # Get the days since last promotion (should be same for all records of a person)
        "#_of_days_since_last_promotion_final",
        min("#_of_days_since_last_promotion").over(window_spec_person)
    ).withColumn(
        # Get average days (should be same for all records of a person)
        "avg_days_between_promotion_final",
        max("avg_days_between_promotion").over(window_spec_person)
    )
    
    # Step 5: Get one record per person with final aggregated features
    final_features = features_aggregated.filter(
        col("vantage_date") == col("latest_vantage_date")
    ).select(
        "person_composite_id",
        col("latest_vantage_date").alias("vantage_date"),
        col("promoted_in_last_1y_final").alias("promoted_in_last_1y"),
        col("promoted_in_last_2y_final").alias("promoted_in_last_2y"),
        col("promoted_in_last_3y_final").alias("promoted_in_last_3y"),
        col("demoted_in_last_1y_final").alias("demoted_in_last_1y"),
        col("demoted_in_last_2y_final").alias("demoted_in_last_2y"),
        col("demoted_in_last_3y_final").alias("demoted_in_last_3y"),
        col("#_promotions_in_last_1y_final").alias("#_promotions_in_last_1y"),
        col("#_promotions_in_last_2y_final").alias("#_promotions_in_last_2y"),
        col("#_promotions_in_last_3y_final").alias("#_promotions_in_last_3y"),
        col("#_demotions_in_last_1y_final").alias("#_demotions_in_last_1y"),
        col("#_demotions_in_last_2y_final").alias("#_demotions_in_last_2y"),
        col("#_demotions_in_last_3y_final").alias("#_demotions_in_last_3y"),
        col("#_of_days_since_last_promotion_final").alias("#_of_days_since_last_promotion"),
        col("avg_days_between_promotion_final").alias("avg_days_between_promotion")
    ).distinct()
    
    # Cache final results
    final_features.cache()
    
    # Unpersist intermediate cached dataframes to free memory
    df_filtered.unpersist()
    
    return final_features

def validate_features(features_df):
    """
    Comprehensive validation of engineered features from multiple perspectives
    
    Parameters:
    features_df: DataFrame with engineered features
    """
    
    print("=" * 60)
    print("COMPREHENSIVE FEATURE VALIDATION")
    print("=" * 60)
    
    total_records = features_df.count()
    unique_employees = features_df.select('person_composite_id').distinct().count()
    
    # Validation 1: Record Uniqueness
    print(f"✓ VALIDATION 1 - RECORD UNIQUENESS:")
    print(f"  Total records: {total_records}")
    print(f"  Unique employees: {unique_employees}")
    if total_records == unique_employees:
        print("  ✓ PASS: Each employee has exactly one record")
    else:
        print("  ✗ FAIL: Duplicate records found for some employees")
    print()
    
    # Validation 2: Binary Feature Values
    print("✓ VALIDATION 2 - BINARY FEATURE VALUES:")
    binary_features = ["promoted_in_last_1y", "promoted_in_last_2y", "promoted_in_last_3y",
                      "demoted_in_last_1y", "demoted_in_last_2y", "demoted_in_last_3y"]
    
    all_binary_valid = True
    for feature in binary_features:
        invalid_count = features_df.filter(~col(feature).isin([0, 1])).count()
        if invalid_count == 0:
            print(f"  ✓ {feature}: All values are 0 or 1")
        else:
            print(f"  ✗ {feature}: {invalid_count} invalid values found")
            all_binary_valid = False
    
    if all_binary_valid:
        print("  ✓ PASS: All binary features contain only 0 and 1 values")
    else:
        print("  ✗ FAIL: Some binary features contain invalid values")
    print()
    
    # Validation 3: Count Feature Non-Negativity
    print("✓ VALIDATION 3 - COUNT FEATURE NON-NEGATIVITY:")
    count_features = ["#_promotions_in_last_1y", "#_promotions_in_last_2y", "#_promotions_in_last_3y",
                     "#_demotions_in_last_1y", "#_demotions_in_last_2y", "#_demotions_in_last_3y"]
    
    all_counts_valid = True
    for feature in count_features:
        negative_count = features_df.filter(col(feature) < 0).count()
        if negative_count == 0:
            print(f"  ✓ {feature}: No negative values")
        else:
            print(f"  ✗ {feature}: {negative_count} negative values found")
            all_counts_valid = False
    
    if all_counts_valid:
        print("  ✓ PASS: All count features are non-negative")
    else:
        print("  ✗ FAIL: Some count features have negative values")
    print()
    
    # Validation 4: Temporal Consistency (3y >= 2y >= 1y)
    print("✓ VALIDATION 4 - TEMPORAL CONSISTENCY:")
    
    # Check promotion count consistency
    promo_inconsistent = features_df.filter(
        (col("#_promotions_in_last_3y") < col("#_promotions_in_last_2y")) |
        (col("#_promotions_in_last_2y") < col("#_promotions_in_last_1y"))
    ).count()
    
    # Check demotion count consistency
    demo_inconsistent = features_df.filter(
        (col("#_demotions_in_last_3y") < col("#_demotions_in_last_2y")) |
        (col("#_demotions_in_last_2y") < col("#_demotions_in_last_1y"))
    ).count()
    
    # Check binary flag consistency for promotions
    promo_binary_inconsistent = features_df.filter(
        (col("promoted_in_last_3y") < col("promoted_in_last_2y")) |
        (col("promoted_in_last_2y") < col("promoted_in_last_1y"))
    ).count()
    
    # Check binary flag consistency for demotions
    demo_binary_inconsistent = features_df.filter(
        (col("demoted_in_last_3y") < col("demoted_in_last_2y")) |
        (col("demoted_in_last_2y") < col("demoted_in_last_1y"))
    ).count()
    
    temporal_issues = promo_inconsistent + demo_inconsistent + promo_binary_inconsistent + demo_binary_inconsistent
    
    if temporal_issues == 0:
        print("  ✓ PASS: All temporal features follow 3y >= 2y >= 1y pattern")
    else:
        print(f"  ✗ FAIL: {temporal_issues} records have temporal inconsistencies")
        print(f"    - Promotion count inconsistencies: {promo_inconsistent}")
        print(f"    - Demotion count inconsistencies: {demo_inconsistent}")
        print(f"    - Promotion binary inconsistencies: {promo_binary_inconsistent}")
        print(f"    - Demotion binary inconsistencies: {demo_binary_inconsistent}")
    print()
    
    # Validation 5: Binary-Count Relationship
    print("✓ VALIDATION 5 - BINARY-COUNT RELATIONSHIP:")
    
    # If binary flag is 1, count should be >= 1; if binary flag is 0, count should be 0
    binary_count_issues = 0
    
    time_periods = ["1y", "2y", "3y"]
    event_types = [("promoted", "promotions"), ("demoted", "demotions")]
    
    for period in time_periods:
        for event_type, count_type in event_types:
            binary_col = f"{event_type}_in_last_{period}"
            count_col = f"#_{count_type}_in_last_{period}"
            
            # Case 1: Binary = 1 but Count = 0
            case1_issues = features_df.filter(
                (col(binary_col) == 1) & (col(count_col) == 0)
            ).count()
            
            # Case 2: Binary = 0 but Count > 0
            case2_issues = features_df.filter(
                (col(binary_col) == 0) & (col(count_col) > 0)
            ).count()
            
            period_issues = case1_issues + case2_issues
            binary_count_issues += period_issues
            
            if period_issues == 0:
                print(f"  ✓ {binary_col} <-> {count_col}: Consistent")
            else:
                print(f"  ✗ {binary_col} <-> {count_col}: {period_issues} inconsistencies")
    
    if binary_count_issues == 0:
        print("  ✓ PASS: All binary flags are consistent with their count counterparts")
    else:
        print(f"  ✗ FAIL: {binary_count_issues} binary-count inconsistencies found")
    print()
    
    # **LINES 237-247: UPDATED VALIDATION FOR DAYS SINCE LAST PROMOTION**
    print("✓ VALIDATION 6 - DAYS SINCE LAST PROMOTION LOGIC:")
    
    # Should be null only when both lst_promo_dt is null AND no promotion events exist
    # or when promotion dates are after vantage_date
    days_logic_issues = features_df.filter(
        # Case 1: days is negative (should never happen)
        (col("#_of_days_since_last_promotion") < 0)
    ).count()
    
    # Additional check: if days is null, verify no valid promotion dates exist
    null_days_with_valid_promos = features_df.filter(
        (col("#_of_days_since_last_promotion").isNull()) &
        (col("#_promotions_in_last_3y") > 0)  # Has promotions but days is null
    ).count()
    
    total_days_issues = days_logic_issues + null_days_with_valid_promos
    
    if total_days_issues == 0:
        print("  ✓ PASS: Days since last promotion logic is correct")
    else:
        print(f"  ✗ FAIL: {total_days_issues} records have incorrect days since last promotion")
        print(f"    - Negative days: {days_logic_issues}")
        print(f"    - Null days with valid promotions: {null_days_with_valid_promos}")
    print()
    
    # Validation 7: Average Days Between Promotions Logic
    print("✓ VALIDATION 7 - AVERAGE DAYS BETWEEN PROMOTIONS LOGIC:")
    
    # Should be 0 for employees with <= 1 promotion, and >= 0 for all
    avg_days_issues = features_df.filter(
        (col("avg_days_between_promotion") < 0)
    ).count()
    
    # Check if employees with 0 or 1 promotions have avg_days = 0
    zero_one_promo_check = features_df.filter(
        (col("#_promotions_in_last_3y") <= 1) & 
        (col("avg_days_between_promotion") != 0)
    ).count()
    
    total_avg_issues = avg_days_issues + zero_one_promo_check
    
    if total_avg_issues == 0:
        print("  ✓ PASS: Average days between promotions logic is correct")
    else:
        print(f"  ✗ FAIL: {total_avg_issues} records have incorrect average days logic")
        print(f"    - Negative average days: {avg_days_issues}")
        print(f"    - Non-zero average for ≤1 promotions: {zero_one_promo_check}")
    print()
    
    # Validation 8: Data Completeness
    print("✓ VALIDATION 8 - DATA COMPLETENESS:")
    
    required_columns = ["person_composite_id", "vantage_date"]
    completeness_issues = 0
    
    for col_name in required_columns:
        null_count = features_df.filter(col(col_name).isNull()).count()
        if null_count == 0:
            print(f"  ✓ {col_name}: No null values")
        else:
            print(f"  ✗ {col_name}: {null_count} null values found")
            completeness_issues += null_count
    
    if completeness_issues == 0:
        print("  ✓ PASS: All required fields are complete")
    else:
        print(f"  ✗ FAIL: {completeness_issues} null values in required fields")
    print()
    
    # Validation 9: Feature Range Validation
    print("✓ VALIDATION 9 - FEATURE RANGE VALIDATION:")
    
    # Days since last promotion should be reasonable (not more than 50 years = ~18250 days)
    unreasonable_days = features_df.filter(
        col("#_of_days_since_last_promotion") > 18250
    ).count()
    
    # Average days should be reasonable (not more than 20 years = ~7300 days)
    unreasonable_avg = features_df.filter(
        col("avg_days_between_promotion") > 7300
    ).count()
    
    range_issues = unreasonable_days + unreasonable_avg
    
    if range_issues == 0:
        print("  ✓ PASS: All feature values are within reasonable ranges")
    else:
        print(f"  ✗ FAIL: {range_issues} records have unreasonable feature values")
        print(f"    - Unreasonable days since last promotion: {unreasonable_days}")
        print(f"    - Unreasonable average days: {unreasonable_avg}")
    print()
    
    # Overall Validation Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    validation_results = [
        total_records == unique_employees,  # Uniqueness
        all_binary_valid,                   # Binary values
        all_counts_valid,                   # Non-negative counts
        temporal_issues == 0,               # Temporal consistency
        binary_count_issues == 0,           # Binary-count relationship
        total_days_issues == 0,             # Days logic (updated variable name)
        total_avg_issues == 0,              # Average days logic
        completeness_issues == 0,           # Completeness
        range_issues == 0                   # Range validation
    ]
    
    passed_validations = sum(validation_results)
    total_validations = len(validation_results)
    
    print(f"Validations Passed: {passed_validations}/{total_validations}")
    
    if passed_validations == total_validations:
        print("🎉 ALL VALIDATIONS PASSED - FEATURES ARE READY FOR USE!")
    else:
        print(f"⚠️  {total_validations - passed_validations} VALIDATION(S) FAILED - PLEASE REVIEW DATA")
    
    print("=" * 60)

def main(input_df):
    """
    Main execution function
    
    Parameters:
    input_df: Input PySpark DataFrame with employee data
    
    Returns:
    DataFrame with engineered features (one record per employee)
    """
    
    try:
        # Create features
        features_df = create_employee_features(input_df)
        
        # Validate features
        validate_features(features_df)
        
        return features_df
        
    except Exception as e:
        print(f"Error in feature engineering pipeline: {str(e)}")
        raise e