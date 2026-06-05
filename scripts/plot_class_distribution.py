import pandas as pd
import plotly.graph_objects as go
import os

# Define colors
AT_RISK_COLOR = '#ED7D31'
TYPICAL_COLOR = '#5B9BD5'

def plot_class_distribution(df, title, output_dir='figures'):
    """
    Plots the distribution of classes in a given DataFrame.
    
    Parameters:
    - df: pandas.DataFrame containing the data
    - title: The title of the plot
    - output_dir: The directory to save the plot image
    """
    class_counts = df['Label'].value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=class_counts.index,
        y=class_counts.values,
        marker_color=[TYPICAL_COLOR if c == 'Typical' else AT_RISK_COLOR for c in class_counts.index]
    )])
    
    fig.update_layout(
        title_text=title,
        xaxis_title="Class",
        yaxis_title="Frequency",
        template="plotly_white"
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fig.write_image(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"))
    fig.show()

if __name__ == "__main__":
    # Load the datasets
    try:
        train_df = pd.read_csv('../datasets/processed/cleaned_train.csv')
        val_df = pd.read_csv('../datasets/processed/cleaned_val.csv')
        test_df = pd.read_csv('../datasets/processed/cleaned_test.csv')

        # Plot distributions
        plot_class_distribution(train_df, "Train Set Class Distribution")
        plot_class_distribution(val_df, "Validation Set Class Distribution")
        plot_class_distribution(test_df, "Test Set Class Distribution")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the cleaned CSV files are in the 'datasets/processed' directory.")

