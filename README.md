# Employee Turnover Survival Analysis with XGBSEKaplanNeighbors

## 📋 Project Overview

This project implements a comprehensive survival analysis framework for predicting employee turnover using the XGBSEKaplanNeighbors model. Unlike traditional binary classification approaches that only predict if an employee will leave, this survival analysis approach predicts **when** they are likely to leave and provides time-dependent retention probabilities.

## 🎯 Business Problem

Employee turnover is costly for organizations, with replacement costs ranging from 50% to 200% of an employee's annual salary. Traditional approaches to turnover prediction often fall short because they:

- Only predict binary outcomes (stay/leave)
- Don't account for timing of departures
- Ignore censored data (employees who haven't left yet)
- Provide limited actionable insights for retention strategies

## 🔬 Solution Approach

### Survival Analysis Framework

Our solution uses **survival analysis** to model employee tenure, which offers several advantages:

- **Time-to-event modeling**: Predicts not just if, but when employees will leave
- **Handles censoring**: Properly accounts for employees still employed at study end
- **Risk stratification**: Identifies high, medium, and low-risk employee groups
- **Time-dependent insights**: Shows how turnover risk changes over time

### XGBSEKaplanNeighbors Model

We implement the XGBSEKaplanNeighbors approach that combines:

- **XGBoost**: Captures complex feature interactions and non-linear patterns
- **Kaplan-Meier Estimation**: Provides smooth, realistic survival curves
- **Neighborhood-based approach**: Finds similar employees for localized survival estimation

## 📊 Dataset Description

### Synthetic Employee Data (n=2,000)

Our model uses comprehensive employee data including:

#### Categorical Features
- **Department**: Engineering, Sales, Marketing, HR, Finance, Operations
- **Job Level**: Junior, Mid, Senior, Lead, Manager
- **Education**: High School, Bachelor, Master, PhD
- **Performance Rating**: Poor, Below Average, Average, Above Average, Excellent
- **Remote Work**: Yes/No
- **Overtime**: Yes/No

#### Continuous Features
- **Age**: Employee age (22-65 years)
- **Salary**: Annual compensation ($30K-$200K)
- **Years at Company**: Tenure length (0.1-20 years)
- **Job Satisfaction**: Rating scale 1-5
- **Work-Life Balance**: Rating scale 1-5
- **Training Hours**: Annual training received (0-100 hours)
- **Commute Distance**: Distance from home (1-50 miles)
- **Team Size**: Number of team members (2-25 people)

#### Target Variables
- **Duration**: Time until departure or censoring (months)
- **Event**: Binary indicator (1=left, 0=still employed)

## 🔧 Technical Implementation

### Dependencies

```bash
pip install xgbse lifelines scikit-learn matplotlib seaborn pandas numpy
```

### Key Functions

#### 1. Data Generation
```python
generate_employee_data(n_samples=2000)
```
- Creates realistic synthetic employee records
- Implements risk-based survival time generation
- Includes appropriate censoring mechanisms

#### 2. Data Preprocessing
```python
preprocess_data(df)
```
- Encodes categorical variables using LabelEncoder
- Prepares data for XGBSEKaplanNeighbors format
- Maintains encoder mappings for interpretability

#### 3. Model Training
```python
fit_xgbse_model(X_train, y_train, X_test, y_test)
```
- Converts data to structured arrays required by XGBSE
- Configures XGBSEKaplanNeighbors with optimal hyperparameters
- Implements early stopping to prevent overfitting

#### 4. Survival Curve Prediction
```python
plot_survival_curves(model, X_train, y_train, X_test, y_test)
```
- Generates individual employee survival curves
- Compares model predictions with Kaplan-Meier baseline
- Creates risk group stratification
- Provides model calibration assessment

#### 5. Model Evaluation
```python
evaluate_model(model, X_train, y_train, X_test, y_test)
```
- Calculates Concordance Index (C-index) for ranking performance
- Computes Brier scores for calibration assessment
- Provides comprehensive performance metrics

## 📈 Model Performance Metrics

### Primary Metrics

1. **Concordance Index (C-index)**
   - Measures ability to correctly rank employee departure times
   - Range: 0.5 (random) to 1.0 (perfect)
   - Target: >0.70 for good performance

2. **Brier Score**
   - Evaluates calibration at specific time points
   - Range: 0.0 (perfect) to 1.0 (worst)
   - Lower values indicate better calibration

3. **Feature Importance**
   - Identifies key drivers of employee turnover
   - Based on XGBoost's tree-based importance measures

## 📊 Visualization Outputs

### 1. Individual Survival Curves
- Shows probability of retention over time for sample employees
- Enables individual risk assessment and personalized interventions

### 2. Model vs. Reality Comparison
- Compares average predicted survival with Kaplan-Meier baseline
- Validates model's ability to capture overall survival patterns

### 3. Risk Group Stratification
- Segments employees into High, Medium, and Low risk categories
- Enables targeted retention strategies for different risk levels

### 4. Model Calibration Plots
- Assesses whether predicted probabilities match observed outcomes
- Critical for trustworthy probability estimates

### 5. Feature Importance Analysis
- Identifies which employee characteristics most influence turnover
- Guides HR policy and intervention strategies

## 💼 Business Applications

### Proactive Retention Management
- **Early Warning System**: Identify at-risk employees before they decide to leave
- **Intervention Timing**: Determine optimal timing for retention conversations
- **Resource Allocation**: Focus retention efforts on highest-risk employees

### Strategic Workforce Planning
- **Departure Forecasting**: Predict when positions will need to be filled
- **Succession Planning**: Identify critical roles at risk
- **Budget Planning**: Anticipate recruitment and training costs

### HR Policy Optimization
- **Root Cause Analysis**: Understand key drivers of employee turnover
- **Policy Impact Assessment**: Measure effects of HR policy changes
- **Benchmarking**: Compare retention performance across departments/teams

## 🔍 Key Insights and Findings

### Risk Factors Identified
Based on the survival analysis, key turnover risk factors typically include:

- **Low job satisfaction scores**
- **Poor work-life balance ratings**
- **Below-average performance ratings**
- **High overtime requirements**
- **Long commute distances**
- **Limited training opportunities**

### Protective Factors
Factors associated with lower turnover risk:

- **Higher compensation levels**
- **Senior positions and tenure**
- **Strong performance ratings**
- **Flexible work arrangements**
- **Regular training and development**

## 🚀 Getting Started

### Quick Start Guide

1. **Install Dependencies**
```bash
pip install xgbse lifelines scikit-learn matplotlib seaborn pandas numpy
```

2. **Run the Analysis**
```python
python employee_survival_analysis.py
```

3. **Interpret Results**
- Review console output for performance metrics
- Examine generated plots for visual insights
- Analyze feature importance for business insights

### Customization Options

- **Sample Size**: Modify `n_samples` parameter for different dataset sizes
- **Risk Factors**: Adjust risk score calculation in `generate_employee_data()`
- **Model Parameters**: Tune XGBSEKaplanNeighbors hyperparameters
- **Visualization**: Customize plot parameters and time horizons

## 📋 Model Limitations and Considerations

### Data Requirements
- **Sample Size**: Requires sufficient sample size for reliable survival estimation
- **Event Rate**: Needs adequate number of departure events for model training
- **Feature Quality**: Performance depends on relevant and high-quality input features

### Assumptions
- **Independence**: Assumes employee departures are independent events
- **Stationarity**: Assumes underlying turnover patterns remain relatively stable
- **Missing Data**: Current implementation assumes complete case analysis

### Validation Considerations
- **Temporal Validation**: Consider time-based train/test splits for real deployment
- **External Validation**: Test on data from different time periods or locations
- **Fairness Assessment**: Evaluate for potential bias across demographic groups

## 🔮 Future Enhancements

### Technical Improvements
- **Deep Learning Integration**: Explore DeepSurv or other neural survival models
- **Time-Varying Covariates**: Handle features that change over time
- **Competing Risks**: Model different types of departures (voluntary vs. involuntary)

### Business Extensions
- **Real-time Scoring**: Implement online prediction system
- **Intervention Modeling**: Assess impact of retention interventions
- **Cost-Benefit Analysis**: Integrate economic impact of turnover predictions

### Data Enhancements
- **External Data**: Incorporate market conditions, economic indicators
- **Behavioral Data**: Include email patterns, system usage, collaboration metrics
- **Survey Integration**: Regular pulse surveys for dynamic satisfaction measures

## 📚 References and Resources

### Academic Literature
- Kalbfleisch, J.D. and Prentice, R.L. (2002). The Statistical Analysis of Failure Time Data
- Klein, J.P. and Moeschberger, M.L. (2003). Survival Analysis: Techniques for Censored and Truncated Data

### Technical Documentation
- [XGBSE Documentation](https://loft-br.github.io/xgboost-survival-embeddings/)
- [Lifelines Documentation](https://lifelines.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Business Applications
- Corporate case studies on HR analytics and employee retention
- Workforce planning and talent management best practices

### Extension Opportunities
- Adapt for different industries or organizational contexts
- Integrate with existing HR information systems
- Develop web-based dashboards for business users

---

**Project Status**: Production Ready  
**Last Updated**: July 2025  
