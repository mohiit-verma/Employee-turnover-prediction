# **Industry-Specific Percentile Performance Analysis**

## **Data Structure Reality Check**

Based on your setup, you have:
```python
# Your existing data structure
datasets_raw = {
    'train': df_clean[df_clean['dataset_split'] == 'train'].copy(),
    'val': df_clean[df_clean['dataset_split'] == 'val'].copy(), 
    'oot': df_clean[df_clean['dataset_split'] == 'oot'].copy()  # This is your validation holdout
}

# Your model predictions (from earlier images)
risk_scores = model_engine.predict_risk_scores(datasets_raw['oot'])
# risk_scores has mean=0.77, std=0.19

# Your actual outcomes
df_oot = datasets_raw['oot']
# Contains: survival_time_days, event_indicator_vol (or event_indicator_all)
```

## ** Analysis Implementation**

### **Step 1: Industry-Client Data Preparation**
```python
def prepare_industry_analysis_data(df_oot, risk_scores):
    """
    Prepare analysis dataset from your existing OOT validation data
    """
    # Create analysis dataframe
    analysis_df = df_oot.copy()
    analysis_df['risk_score'] = risk_scores
    
    # You need to identify industries and clients from your existing columns
    # Adjust these column names based on your actual data structure
    required_columns = ['client_id', 'industry', 'naics_2digit']  # or whatever you have
    
    available_columns = analysis_df.columns.tolist()
    print("Available columns:", available_columns[:20])  # Check what you actually have
    
    # Create industry mapping if needed
    if 'industry' not in analysis_df.columns and 'naics_2digit' in analysis_df.columns:
        industry_mapping = {
            '44': 'retail', '45': 'retail',  # Retail trade
            '31': 'manufacturing', '32': 'manufacturing', '33': 'manufacturing',
            '54': 'professional_services',  # Professional services
            '62': 'healthcare'  # Healthcare and social assistance
        }
        analysis_df['industry'] = analysis_df['naics_2digit'].astype(str).map(industry_mapping)
    
    # Filter out rows without industry classification
    analysis_df = analysis_df.dropna(subset=['industry'])
    
    return analysis_df

# Prepare your data
df_analysis = prepare_industry_analysis_data(datasets_raw['oot'], risk_scores)
print(f"Analysis dataset: {len(df_analysis):,} employees")
print("Industry distribution:")
print(df_analysis['industry'].value_counts())
```

### **Step 2: Client Selection Within Industries**
```python
def select_analysis_clients(df_analysis, min_employees_per_client=500):
    """
    Select clients with sufficient sample sizes for reliable analysis
    """
    # Get client sizes by industry
    client_sizes = df_analysis.groupby(['industry', 'client_id']).size().reset_index(name='employee_count')
    
    # Filter clients with sufficient data
    valid_clients = client_sizes[client_sizes['employee_count'] >= min_employees_per_client]
    
    analysis_targets = {}
    for industry in valid_clients['industry'].unique():
        industry_clients = valid_clients[valid_clients['industry'] == industry]['client_id'].tolist()
        analysis_targets[industry] = industry_clients[:5]  # Max 5 clients per industry
        
        print(f"{industry}: {len(industry_clients)} valid clients, analyzing top 5")
        for client in analysis_targets[industry]:
            client_size = client_sizes[
                (client_sizes['industry'] == industry) & 
                (client_sizes['client_id'] == client)
            ]['employee_count'].iloc[0]
            print(f"  - {client}: {client_size:,} employees")
    
    return analysis_targets

analysis_targets = select_analysis_clients(df_analysis)
```

### **Step 3:  Performance Analysis**
```python
def analyze_client_percentile_performance(df_client, 
                                        time_horizons=[90, 180, 270, 365],
                                        percentile_thresholds=[5, 10, 15, 20, 25]):
    """
    Analyze performance for a single client using your actual data structure
    """
    results = []
    
    # Use your actual column names
    survival_time_col = 'survival_time_days' 
    event_col = 'event_indicator_vol'  # or 'event_indicator_all'
    risk_score_col = 'risk_score'
    
    for horizon in time_horizons:
        # Create binary outcome: did they leave within this horizon?
        y_true = ((df_client[survival_time_col] <= horizon) & 
                 (df_client[event_col] == 1)).astype(int)
        
        base_event_rate = y_true.mean()
        total_events = y_true.sum()
        
        if total_events < 10:  # Skip horizons with too few events
            continue
            
        for percentile in percentile_thresholds:
            # Define threshold: top X% riskiest employees
            threshold = np.percentile(df_client[risk_score_col], 100 - percentile)
            
            # Predict: 1 if in top percentile, 0 otherwise  
            y_pred = (df_client[risk_score_col] >= threshold).astype(int)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Business metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            event_capture_rate = tp / total_events if total_events > 0 else 0
            
            # Population flagged (should be close to percentile)
            population_flagged_pct = (tp + fp) / len(df_client) * 100
            
            results.append({
                'time_horizon_days': horizon,
                'percentile_threshold': percentile,
                'risk_score_threshold': threshold,
                'true_positives': tp,
                'false_positives': fp, 
                'true_negatives': tn,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'event_capture_rate': event_capture_rate,
                'population_flagged_pct': population_flagged_pct,
                'base_event_rate_pct': base_event_rate * 100,
                'total_events': total_events,
                'total_employees': len(df_client)
            })
    
    return pd.DataFrame(results)
```

### **Step 4: Run Analysis on Your Actual Data**
```python
def run_industry_analysis_(df_analysis, analysis_targets):
    """
    Run analysis using your actual data structure
    """
    all_results = {}
    
    for industry, client_list in analysis_targets.items():
        print(f"\n=== ANALYZING {industry.upper()} INDUSTRY ===")
        industry_results = []
        
        for client_id in client_list:
            # Filter to specific client
            client_data = df_analysis[df_analysis['client_id'] == client_id].copy()
            
            if len(client_data) < 500:
                print(f"Skipping {client_id}: {len(client_data)} employees (too small)")
                continue
            
            event_rate = client_data['event_indicator_vol'].mean()  # or event_indicator_all
            print(f"Analyzing {client_id}: {len(client_data):,} employees, {event_rate:.1%} event rate")
            
            try:
                # Run percentile analysis
                client_results = analyze_client_percentile_performance(client_data)
                
                if not client_results.empty:
                    client_results['client_id'] = client_id
                    client_results['industry'] = industry
                    industry_results.append(client_results)
                    
            except Exception as e:
                print(f"Error analyzing {client_id}: {e}")
                continue
        
        # Combine results for this industry
        if industry_results:
            all_results[industry] = pd.concat(industry_results, ignore_index=True)
            print(f"Successfully analyzed {len(industry_results)} {industry} clients")
        
    return all_results

# Run the  analysis
all_results = run_industry_analysis_(df_analysis, analysis_targets)
```

### **Step 5: Business Insights Generation**
```python
def generate_percentile_recommendations(all_results):
    """
    Generate specific percentile recommendations by industry
    """
    recommendations = {}
    
    for industry, results in all_results.items():
        if results.empty:
            continue
            
        print(f"\n=== {industry.upper()} INDUSTRY RECOMMENDATIONS ===")
        
        # Find best performing configurations
        best_configs = []
        
        # Scenario 1: Resource constrained (10% max)
        constrained = results[results['population_flagged_pct'] <= 12]
        if not constrained.empty:
            best_constrained = constrained.loc[constrained['event_capture_rate'].idxmax()]
            best_configs.append({
                'scenario': 'Resource Constrained (≤10% workforce)',
                'percentile': int(best_constrained['percentile_threshold']),
                'horizon': int(best_constrained['time_horizon_days']),
                'event_capture': f"{best_constrained['event_capture_rate']:.1%}",
                'precision': f"{best_constrained['precision']:.1%}",
                'employees_flagged': f"{best_constrained['population_flagged_pct']:.1f}%"
            })
        
        # Scenario 2: Balanced approach (15-20%)
        balanced = results[
            (results['population_flagged_pct'] >= 13) & 
            (results['population_flagged_pct'] <= 22)
        ]
        if not balanced.empty:
            best_balanced = balanced.loc[balanced['event_capture_rate'].idxmax()]
            best_configs.append({
                'scenario': 'Balanced Approach (15-20% workforce)',
                'percentile': int(best_balanced['percentile_threshold']),
                'horizon': int(best_balanced['time_horizon_days']),
                'event_capture': f"{best_balanced['event_capture_rate']:.1%}",
                'precision': f"{best_balanced['precision']:.1%}",
                'employees_flagged': f"{best_balanced['population_flagged_pct']:.1f}%"
            })
        
        # Print recommendations
        for config in best_configs:
            print(f"\n{config['scenario']}:")
            print(f"  Recommended: Top {config['percentile']}% over {config['horizon']} days")
            print(f"  Performance: {config['event_capture']} event capture, {config['precision']} precision")
            print(f"  Impact: {config['employees_flagged']} of workforce flagged")
        
        recommendations[industry] = best_configs
        
        # Industry-specific insights
        avg_base_rate = results['base_event_rate_pct'].mean()
        print(f"\nIndustry Insights:")
        print(f"  Average baseline turnover: {avg_base_rate:.1f}%")
        
        if avg_base_rate > 20:
            print("  → High turnover industry: Consider 15-20% thresholds")
        elif avg_base_rate < 10:
            print("  → Low turnover industry: Consider 10-12% thresholds")
        else:
            print("  → Moderate turnover industry: Consider 12-15% thresholds")
    
    return recommendations

# Generate recommendations
recommendations = generate_percentile_recommendations(all_results)
```

### **Key Corrections Made:**

1. **Data Source**: Uses your actual `datasets_raw['oot']` validation holdout
2. **Column Names**: References your actual columns (`survival_time_days`, `event_indicator_vol`)
3. **Risk Scores**: Uses your computed `risk_scores` from the model
4. **Industry Mapping**: Works with your available client/industry identifiers
5. **Realistic Constraints**: Accounts for minimum sample sizes and data availability

### **What to Check First:**
```python
# Before running the full analysis, check your data structure
print("OOT dataset shape:", datasets_raw['oot'].shape)
print("Available columns:", datasets_raw['oot'].columns.tolist())
print("Risk scores shape:", risk_scores.shape)

# Check for industry/client identifiers
potential_industry_cols = ['naics_2digit', 'industry', 'sector']
potential_client_cols = ['client_id', 'company_id', 'org_id']

for col in potential_industry_cols:
    if col in datasets_raw['oot'].columns:
        print(f"Found industry column: {col}")
        print(datasets_raw['oot'][col].value_counts().head())

for col in potential_client_cols:
    if col in datasets_raw['oot'].columns:
        print(f"Found client column: {col}")
        print(f"Unique clients: {datasets_raw['oot'][col].nunique()}")
```

This  approach works with your actual data structure and will provide the industry-specific percentile performance analysis needed to support the business case for percentile-based risk categories.