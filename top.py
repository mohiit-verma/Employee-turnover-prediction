import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.plotting import plot_lifetimes
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_employee_data(n_samples=2000):
    """
    Generate synthetic employee turnover data for survival analysis
    """
    # Categorical features
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']
    education = ['Bachelor', 'Master', 'PhD', 'High School']
    performance_ratings = ['Poor', 'Below Average', 'Average', 'Above Average', 'Excellent']
    
    data = {
        # Categorical features
        'department': np.random.choice(departments, n_samples, 
                                     p=[0.25, 0.20, 0.15, 0.10, 0.15, 0.15]),
        'job_level': np.random.choice(job_levels, n_samples,
                                    p=[0.30, 0.25, 0.25, 0.15, 0.05]),
        'education': np.random.choice(education, n_samples,
                                    p=[0.40, 0.35, 0.15, 0.10]),
        'performance_rating': np.random.choice(performance_ratings, n_samples,
                                             p=[0.05, 0.15, 0.40, 0.30, 0.10]),
        'remote_work': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'overtime': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        
        # Continuous features
        'age': np.random.normal(35, 8, n_samples).clip(22, 65),
        'salary': np.random.normal(75000, 25000, n_samples).clip(30000, 200000),
        'years_at_company': np.random.exponential(3, n_samples).clip(0.1, 20),
        'job_satisfaction': np.random.normal(3.5, 1.2, n_samples).clip(1, 5),
        'work_life_balance': np.random.normal(3.2, 1.1, n_samples).clip(1, 5),
        'training_hours': np.random.gamma(2, 10, n_samples).clip(0, 100),
        'commute_distance': np.random.exponential(15, n_samples).clip(1, 50),
        'team_size': np.random.poisson(8, n_samples).clip(2, 25)
    }
    
    df = pd.DataFrame(data)
    
    # Create survival times and censoring based on features
    # Higher risk factors lead to shorter survival times
    risk_score = (
        -0.1 * (df['salary'] / 10000) +  # Higher salary = lower risk
        -0.3 * df['job_satisfaction'] +   # Higher satisfaction = lower risk
        -0.2 * df['work_life_balance'] +  # Better balance = lower risk
        0.05 * df['commute_distance'] +   # Longer commute = higher risk
        0.5 * (df['performance_rating'] == 'Poor').astype(int) +
        0.3 * (df['overtime'] == 'Yes').astype(int) +
        -0.1 * df['training_hours'] / 10 +  # More training = lower risk
        0.02 * df['age'] +  # Slight age effect
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Generate survival times using exponential distribution
    # Higher risk_score leads to shorter survival times
    baseline_hazard = 0.08
    survival_times = np.random.exponential(1 / (baseline_hazard * np.exp(risk_score)))
    
    # Set observation period (e.g., 5 years = 60 months)
    observation_period = 60
    
    # Create censoring: event occurs if survival time < observation period
    df['duration'] = np.minimum(survival_times, observation_period)
    df['event'] = (survival_times <= observation_period).astype(int)
    
    # Round duration to months
    df['duration'] = np.round(df['duration'], 1)
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for XGBSEKaplanNeighbors model
    """
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['department', 'job_level', 'education', 'performance_rating', 'remote_work', 'overtime']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    return df_processed, label_encoders

def fit_xgbse_model(X_train, y_train, X_test, y_test):
    """
    Fit XGBSEKaplanNeighbors model
    """
    # Convert to structured array format required by XGBSE
    y_train_structured = convert_to_structured(y_train['duration'], y_train['event'])
    y_test_structured = convert_to_structured(y_test['duration'], y_test['event'])
    
    # Initialize XGBSEKaplanNeighbors model
    model = XGBSEKaplanNeighbors(
        # XGBoost parameters
        xgb_params={'n_estimators':200,
        'max_depth':6,
        'learning_rate':0.1,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'random_state':42,
        'verbose':True},
        # Kaplan neighbors parameters
        n_neighbors=50,  # Number of neighbors for Kaplan-Meier estimation
        radius=None,     # Use k-nearest neighbors instead of radius
    )
    
    # Fit the model
    print("Fitting XGBSEKaplanNeighbors model...")
    model.fit(
        X_train, y_train_structured,
        validation_data=[X_test, y_test_structured],
        early_stopping_rounds=20,
        # verbose=False
    )
    
    return model

def plot_survival_curves(model: XGBSEKaplanNeighbors, X_train, y_train, X_test, y_test, sample_employees=5):
    """
    Plot survival curves using XGBSEKaplanNeighbors predictions
    """
    # Time points for prediction
    time_points = np.arange(1, 61, 1)  # 1 to 60 months
    
    # Get survival function predictions
    print("Generating survival predictions...")
    survival_curves = model.predict(X_test, time_bins=time_points)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Sample individual survival curves
    ax1 = axes[0, 0]
    sample_indices = np.random.choice(len(X_test), sample_employees, replace=False)
    
    for i, idx in enumerate(sample_indices):
        ax1.plot(time_points, survival_curves.iloc[idx], 
                label=f'Employee {idx+1}', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('Individual Employee Survival Curves (XGBSEKaplanNeighbors)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Average survival curve vs Kaplan-Meier
    ax2 = axes[0, 1]
    
    # Average survival curve from XGBSE
    avg_survival = survival_curves.mean(axis=0)
    ax2.plot(time_points, avg_survival, label='XGBSEKaplanNeighbors (Average)', 
             linewidth=3, color='red')
    
    # Kaplan-Meier estimator for comparison
    kmf = KaplanMeierFitter()
    kmf.fit(y_test['duration'], y_test['event'])
    
    # Get KM survival function at our time points
    km_survival = kmf.survival_function_at_times(time_points)
    ax2.plot(time_points, km_survival, label='Kaplan-Meier (Actual)', 
             linewidth=3, color='blue', linestyle='--')
    
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Survival Probability')
    ax2.set_title('Average Survival Curve Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Survival curves by risk quantiles
    ax3 = axes[1, 0]
    
    # Calculate risk scores (median survival time)
    median_survival_times = []
    for idx in range(len(survival_curves)):
        curve = survival_curves.iloc[idx]
        # Find median survival time (where survival probability = 0.5)
        try:
            median_time = time_points[np.where(curve <= 0.5)[0][0]]
        except:
            median_time = time_points[-1]  # If never reaches 0.5, use max time
        median_survival_times.append(median_time)
    
    # Create risk groups based on median survival times
    risk_quantiles = pd.qcut(median_survival_times, q=3, labels=['High Risk', 'Medium Risk', 'Low Risk'])
    
    colors = ['red', 'orange', 'green']
    for i, group in enumerate(['High Risk', 'Medium Risk', 'Low Risk']):
        group_indices = np.where(risk_quantiles == group)[0]
        group_curves = survival_curves.iloc[group_indices]
        group_avg = group_curves.mean(axis=0)
        
        ax3.plot(time_points, group_avg, label=f'{group} (n={len(group_indices)})', 
                linewidth=3, color=colors[i])
    
    ax3.set_xlabel('Time (months)')
    ax3.set_ylabel('Survival Probability')
    ax3.set_title('Survival Curves by Risk Groups')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Model calibration plot
    ax4 = axes[1, 1]
    
    # For calibration, we'll look at predicted vs observed survival at specific time points
    calibration_times = [12, 24, 36, 48]
    
    for t in calibration_times:
        if t in time_points:
            t_idx = list(time_points).index(t)
            predicted_surv = survival_curves.iloc[:, t_idx]
            
            # Create bins for predicted survival probabilities
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            observed_surv = []
            for i in range(len(bins)-1):
                mask = (predicted_surv >= bins[i]) & (predicted_surv < bins[i+1])
                if mask.sum() > 0:
                    # Calculate observed survival rate in this bin
                    mask_indices = np.where(mask)[0]  # Convert boolean mask to integer indices
                    bin_data = y_test.iloc[mask_indices]  # Use integer indices with iloc
                    # bin_data = y_test.iloc[mask]
                    obs_rate = ((bin_data['duration'] > t) | (bin_data['event'] == 0)).mean()
                    observed_surv.append(obs_rate)
                else:
                    observed_surv.append(np.nan)
            
            ax4.plot(bin_centers, observed_surv, 'o-', label=f'{t} months', alpha=0.7)
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax4.set_xlabel('Predicted Survival Probability')
    ax4.set_ylabel('Observed Survival Probability')
    ax4.set_title('Model Calibration at Different Time Points')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return survival_curves, median_survival_times

def evaluate_model(model: XGBSEKaplanNeighbors, X_train, y_train, X_test, y_test):
    """
    Evaluate the XGBSEKaplanNeighbors model
    """
    # Convert to structured arrays
    y_train_structured = convert_to_structured(y_train['duration'], y_train['event'])
    y_test_structured = convert_to_structured(y_test['duration'], y_test['event'])
    
    # Get time points for evaluation
    time_points = np.arange(1, 61, 1)
    
    # Predict survival curves
    survival_curves_test = model.predict(X_test, time_bins=time_points)
    survival_curves_train = model.predict(X_train, time_bins=time_points)
    
    # Calculate concordance index using median survival times
    def get_median_survival_time(curve):
        try:
            return time_points[np.where(curve <= 0.5)[0][0]]
        except:
            return time_points[-1]
    
    median_times_test = [get_median_survival_time(survival_curves_test.iloc[i]) 
                        for i in range(len(survival_curves_test))]
    median_times_train = [get_median_survival_time(survival_curves_train.iloc[i]) 
                         for i in range(len(survival_curves_train))]
    
    # Calculate concordance indices
    c_index_test = concordance_index(y_test['duration'], median_times_test, y_test['event'])
    c_index_train = concordance_index(y_train['duration'], median_times_train, y_train['event'])
    
    # Calculate Brier score at specific time points
    brier_scores = []
    for t in [12, 24, 36, 48]:
        if t in time_points:
            t_idx = list(time_points).index(t)
            predicted_surv = survival_curves_test.iloc[:, t_idx]
            
            # Observed survival status at time t
            observed_surv = ((y_test['duration'] > t) | (y_test['event'] == 0)).astype(int)
            
            # Brier score
            brier = mean_squared_error(observed_surv, predicted_surv)
            brier_scores.append(brier)
    
    print("Model Evaluation Results:")
    print("=" * 40)
    print(f"Training Concordance Index: {c_index_train:.4f}")
    print(f"Test Concordance Index: {c_index_test:.4f}")
    print(f"Number of events in test set: {y_test['event'].sum()}")
    print(f"Censoring rate: {(1 - y_test['event'].mean()):.2%}")
    print("\nBrier Scores at different time points:")
    for i, t in enumerate([12, 24, 36, 48]):
        if i < len(brier_scores):
            print(f"  {t} months: {brier_scores[i]:.4f}")
    
    return c_index_test, brier_scores

def plot_feature_importance(model: XGBSEKaplanNeighbors, feature_names):
    """
    Plot feature importance from the XGBoost model within XGBSE
    """
    # Get feature importance from the underlying XGBoost model
    # importance_dict = model.bst.get_score(importance_type='weight')
    importance_dict = model.feature_importances_
    
    # Create DataFrame for plotting
    features = list(importance_dict.keys())
    values = []
    for f in features:
        values.append(importance_dict[f])
    importance_df = pd.DataFrame(
        {'feature': features, 'importance': values}
        ).sort_values('importance', ascending=False)
    
    importance_explanations = {
        'weight': 'Frequency of feature usage in splits',
        'gain': 'Average gain when feature is used for splitting',
        'cover': 'Average coverage when feature is used for splitting', 
        'total_gain': 'Total gain across all splits using the feature',
        'total_cover': 'Total coverage across all splits using the feature'
    }
    importance_type = 'weight'
    print(f"\nFeature Importance Type: {importance_type}")
    print(f"Meaning: {importance_explanations.get(importance_type, 'Custom importance measure')}")
    print("-" * 60)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel(f'Feature Importance ({importance_type})')
    plt.title(f'Top 15 Feature Importances - {importance_type.title()} Method')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(top_features['importance']) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed importance analysis
    print(f"\nTop 10 Features by {importance_type.title()}:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<20} = {row['importance']:>8.1f}")
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance (Weight)')
    plt.title('Top 15 Feature Importances (XGBSEKaplanNeighbors)')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(top_features['importance']) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

# Main execution
if __name__ == "__main__":
    print("Employee Turnover Survival Analysis with XGBSEKaplanNeighbors")
    print("=" * 60)
    
    print("Generating employee turnover data...")
    df = generate_employee_data(2000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Event rate: {df['event'].mean():.2%}")
    print(f"Average duration: {df['duration'].mean():.1f} months")
    print(f"Median duration: {df['duration'].median():.1f} months")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Basic data exploration
    print("\nDataset Summary:")
    print(df.describe())
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    
    # Prepare features and target
    feature_cols = [col for col in df_processed.columns if col not in ['duration', 'event']]
    X = df_processed[feature_cols]
    y = df_processed[['duration', 'event']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y['event']
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training event rate: {y_train['event'].mean():.2%}")
    print(f"Test event rate: {y_test['event'].mean():.2%}")
    
    # Fit XGBSEKaplanNeighbors model
    print("\nTraining XGBSEKaplanNeighbors model...")
    model = fit_xgbse_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    print("\nEvaluating model...")
    c_index, brier_scores = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Plot survival curves
    print("\nGenerating survival curve plots...")
    survival_curves, median_survival_times = plot_survival_curves(
        model, X_train, y_train, X_test, y_test, sample_employees=8
    )
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    importance_df = plot_feature_importance(model, feature_cols)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("=" * 30)
    print(f"Model Performance (C-index): {c_index:.4f}")
    print(f"Average predicted median survival: {np.mean(median_survival_times):.1f} months")
    print(f"Range of predicted survival times: {np.min(median_survival_times):.1f} - {np.max(median_survival_times):.1f} months")
    
    # Risk stratification
    risk_groups = pd.qcut(median_survival_times, q=3, labels=['High', 'Medium', 'Low'])
    risk_distribution = risk_groups.value_counts()
    print(f"\nRisk Group Distribution:")
    for risk, count in risk_distribution.items():
        print(f"  {risk} Risk: {count} employees ({count/len(risk_groups)*100:.1f}%)")
    
    print("\nAnalysis complete! Check the generated plots for detailed insights.")
    print("The model can now be used to predict employee turnover risk and survival curves.")
