def comprehensive_multicollinearity_analysis(df, target_col, corr_threshold=0.8, vif_threshold=5):
    """Complete multicollinearity analysis pipeline"""
    
    print("=== COMPREHENSIVE MULTICOLLINEARITY ANALYSIS ===\n")
    
    # Prepare data
    X = df.drop(target_col, axis=1)
    print(f"Analyzing {X.shape[1]} features across {X.shape[0]} records\n")
    
    # 1. Correlation Analysis
    corr_matrix = X.corr()
    high_corr_pairs = find_high_correlations(corr_matrix, corr_threshold)
    print(f"1. CORRELATION ANALYSIS")
    print(f"   High correlation pairs (>{corr_threshold}): {len(high_corr_pairs)}")
    
    if len(high_corr_pairs) > 0:
        print("   Top 5 highest correlations:")
        print(high_corr_pairs.head().to_string(index=False))
    print()
    
    # 2. VIF Analysis
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    vif_results = calculate_vif(X_scaled)
    high_vif_features = vif_results[vif_results['VIF'] > vif_threshold]
    
    print(f"2. VIF ANALYSIS")
    print(f"   Features with VIF > {vif_threshold}: {len(high_vif_features)}")
    print(f"   Maximum VIF: {vif_results['VIF'].max():.2f}")
    if len(high_vif_features) > 0:
        print("   Top 5 highest VIF features:")
        print(high_vif_features.head().to_string(index=False))
    print()
    
    # 3. Condition Number
    condition_num = condition_number_analysis(corr_matrix)
    print()
    
    # 4. Summary and Recommendations
    print("4. SUMMARY & RECOMMENDATIONS")
    
    multicollinearity_level = "Low"
    if len(high_corr_pairs) > 5 or len(high_vif_features) > 5 or condition_num > 100:
        multicollinearity_level = "High"
    elif len(high_corr_pairs) > 2 or len(high_vif_features) > 2 or condition_num > 30:
        multicollinearity_level = "Moderate"
    
    print(f"   Overall multicollinearity level: {multicollinearity_level}")
    
    if multicollinearity_level in ["Moderate", "High"]:
        print("   Recommendations:")
        print("   - Consider removing highly correlated features")
        print("   - Apply dimensionality reduction (PCA, Factor Analysis)")
        print("   - Use regularized regression models (Ridge, Lasso)")
        print("   - Perform feature selection before modeling")
    
    return {
        'high_correlations': high_corr_pairs,
        'high_vif_features': high_vif_features,
        'condition_number': condition_num,
        'multicollinearity_level': multicollinearity_level
    }

# Run comprehensive analysis
results = comprehensive_multicollinearity_analysis(df, 'your_target_column')