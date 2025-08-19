import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
corr_matrix = df.corr()

# Find high correlations
def find_high_correlations(corr_matrix, threshold=0.8):
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    return pd.DataFrame(high_corr_pairs)

high_corr_df = find_high_correlations(corr_matrix, threshold=0.8)
print(f"High correlation pairs (>0.8): {len(high_corr_df)}")
print(high_corr_df.sort_values('correlation', key=abs, ascending=False))

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Prepare data (exclude target variable)
X = df.drop('target_column', axis=1)  # Replace with your target column name

# Standardize features for VIF calculation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Calculate VIF for each feature
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(df.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

vif_results = calculate_vif(X_scaled_df)
print("VIF Analysis:")
print(vif_results)
print(f"\nFeatures with VIF > 5: {len(vif_results[vif_results['VIF'] > 5])}")
print(f"Features with VIF > 10: {len(vif_results[vif_results['VIF'] > 10])}")

# Calculate condition number
def condition_number_analysis(corr_matrix):
    eigenvalues = np.linalg.eigvals(corr_matrix)
    condition_number = np.max(eigenvalues) / np.min(eigenvalues)
    
    print(f"Condition Number: {condition_number:.2f}")
    print(f"Multicollinearity Assessment:")
    if condition_number < 30:
        print("- Low multicollinearity")
    elif condition_number < 100:
        print("- Moderate multicollinearity") 
    else:
        print("- High multicollinearity")
    
    # Show eigenvalue distribution
    print(f"\nEigenvalue Statistics:")
    print(f"Max eigenvalue: {np.max(eigenvalues):.4f}")
    print(f"Min eigenvalue: {np.min(eigenvalues):.4f}")
    print(f"Smallest 5 eigenvalues: {np.sort(eigenvalues)[:5]}")
    
    return condition_number

condition_num = condition_number_analysis(corr_matrix)


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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """Calculate Cramér's V correlation coefficient"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

def categorical_correlation_plot(df):
    """Create simple correlation heatmap for categorical features"""
    
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create correlation matrix
    n = len(cat_cols)
    corr_matrix = np.zeros((n, n))
    
    for i, col1 in enumerate(cat_cols):
        for j, col2 in enumerate(cat_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = cramers_v(df[col1], df[col2])
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=cat_cols, columns=cat_cols)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title("Categorical Features Correlation (Cramér's V)")
    plt.tight_layout()
    plt.show()
    
    # Print high correlations
    print("High correlations (> 0.5):")
    for i in range(n):
        for j in range(i+1, n):
            if corr_matrix[i, j] > 0.5:
                print(f"{cat_cols[i]} - {cat_cols[j]}: {corr_matrix[i, j]:.3f}")
    
    return corr_df

# Usage
correlation_matrix = categorical_correlation_plot(df)