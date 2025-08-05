import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import ceil

def plot_binary_features_bar_charts(data, feature_columns, figsize=(15, 5), colors=['#3498db', '#e74c3c']):
    """
    Plot bar charts for binary features with category counts displayed on bars.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the binary features
    feature_columns : list
        List of column names to plot (should be binary features)
    figsize : tuple, optional
        Figure size for each row of plots (default: (15, 5))
    colors : list, optional
        List of colors for categories [color_for_0, color_for_1] (default: blue and red)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing all subplots
    """
    
    # Calculate number of rows needed (3 plots per row)
    n_features = len(feature_columns)
    n_rows = ceil(n_features / 3)
    n_cols = min(3, n_features)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    
    # Handle case where there's only one row
    if n_rows == 1:
        if n_cols == 1:
            axes = [axes]
        else:
            axes = [axes]
    else:
        # Flatten axes array for easier iteration
        axes = axes.flatten() if n_features > 1 else [axes]
    
    # Plot each feature
    for i, feature in enumerate(feature_columns):
        ax = axes[i]
        
        # Get value counts for the feature
        counts = data[feature].value_counts().sort_index()
        
        # Create bar plot
        bars = ax.bar(counts.index.astype(str), counts.values, 
                     color=[colors[int(idx)] for idx in counts.index])
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Customize the plot
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Category', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        
        # Set y-axis to start from 0 and add some padding
        ax.set_ylim(0, max(counts.values) * 1.1)
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Hide extra subplots if any
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    return fig

# Example usage function
def example_usage():
    """
    Example of how to use the plot_binary_features_bar_charts function
    """
    # Create sample binary data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'feature_1': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'feature_2': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'feature_3': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'feature_4': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'feature_5': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    })
    
    # List of features to plot
    features_to_plot = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Create the plots
    fig = plot_binary_features_bar_charts(
        data=sample_data, 
        feature_columns=features_to_plot,
        figsize=(15, 5),
        colors=['#2ecc71', '#e67e22']  # Green for 0, Orange for 1
    )
    
    plt.show()
    return fig

# Alternative function with more customization options
def plot_binary_features_advanced(data, feature_columns, figsize=(15, 5), 
                                colors=['#3498db', '#e74c3c'], 
                                category_labels=['No', 'Yes'],
                                show_percentage=False):
    """
    Advanced version with additional customization options.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the binary features
    feature_columns : list
        List of column names to plot (should be binary features)
    figsize : tuple, optional
        Figure size for each row of plots (default: (15, 5))
    colors : list, optional
        List of colors for categories [color_for_0, color_for_1]
    category_labels : list, optional
        Custom labels for categories (default: ['No', 'Yes'])
    show_percentage : bool, optional
        Whether to show percentages along with counts (default: False)
    """
    
    n_features = len(feature_columns)
    n_rows = ceil(n_features / 3)
    n_cols = min(3, n_features)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    
    if n_rows == 1:
        if n_cols == 1:
            axes = [axes]
        else:
            axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(feature_columns):
        ax = axes[i]
        
        counts = data[feature].value_counts().sort_index()
        total = counts.sum()
        
        # Use custom labels if provided
        x_labels = [category_labels[int(idx)] if int(idx) < len(category_labels) else str(idx) 
                   for idx in counts.index]
        
        bars = ax.bar(x_labels, counts.values, 
                     color=[colors[int(idx)] for idx in counts.index])
        
        # Add labels with optional percentages
        for bar, count, idx in zip(bars, counts.values, counts.index):
            height = bar.get_height()
            if show_percentage:
                percentage = (count / total) * 100
                label = f'{count}\n({percentage:.1f}%)'
            else:
                label = f'{count}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Category', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_ylim(0, max(counts.values) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout(pad=3.0)
    return fig