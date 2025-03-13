import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import argparse
import glob

pio.kaleido.scope.mathjax = None

def is_headless():
    """Check if we're running in a headless environment (like a remote server)"""
    return 'SSH_CONNECTION' in os.environ or 'DISPLAY' not in os.environ

def find_metrics_file(base_dir, agent_type, disturbance):
    """Find metrics file with the new naming convention."""
    search_pattern = os.path.join(base_dir, agent_type, f"{disturbance}_*_{agent_type}_*.csv")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"Warning: No matching metrics file found with pattern {search_pattern}")
        return None
    
    # Return the first matching file (there should typically be only one)
    return matching_files[0]

def plot_success_metrics(base_dir=None, save_plots=True, show_plots=False):
    """
    Create bar charts showing Success, Misses, and Hard Misses as percentages for each agent type.
    
    Args:
        base_dir: Directory containing the metrics. If None, uses the script's directory.
        save_plots: Whether to save the plots as HTML files
        show_plots: Whether to display the plots in the browser
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we can show plots
    if show_plots and is_headless():
        print("Warning: Running in headless environment. Cannot show plots in browser.")
        print("Plots will only be saved as HTML files.")
        show_plots = False
        save_plots = True
        
    agent_types = ["planning", "noplan", "truemf"]
    disturbances = ["turb", "gusts", "wind"]
    severity_colors = {
        "off": "#4CAF50",        # Green
        "light": "#2196F3",      # Blue
        "moderate": "#FFC107",   # Yellow/Amber
        "severe": "#F44336"      # Red
    }
    
    # Dictionary to store all figures
    figures = {}
    
    # Process each agent type
    for agent_type in agent_types:
        print(f"Processing plots for {agent_type}...")
        
        # Check if agent type directory exists
        agent_dir = os.path.join(base_dir, agent_type)
        if not os.path.exists(agent_dir):
            print(f"Warning: Directory {agent_dir} does not exist. Skipping.")
            continue
        
        # Create a subplot figure for this agent type (one row per disturbance)
        fig = make_subplots(
            rows=len(disturbances), 
            cols=1,
            subplot_titles=[f"{dist.capitalize()} Disturbance" for dist in disturbances],
            vertical_spacing=0.15
        )
        
        has_data = False  # Flag to check if any data was found
        
        # Dictionary to track which categories have been added to the legend
        legend_added = {}
        
        for i, disturbance in enumerate(disturbances):
            row = i + 1
            
            # Find metrics file with the new naming convention
            metrics_file = find_metrics_file(base_dir, agent_type, disturbance)
            
            if not metrics_file or not os.path.exists(metrics_file):
                print(f"Warning: No metrics file found for {agent_type} with {disturbance}. Skipping.")
                continue
            
            print(f"Using metrics file: {os.path.basename(metrics_file)}")
            has_data = True
            
            # Read the CSV file
            df = pd.read_csv(metrics_file)
            
            # Don't filter out SEM rows - we need them for error bars
            # Extract mean and SEM dataframes
            df_mean = df[df['Metric_Type'] == 'Mean'].copy()  # Create explicit copies
            df_sem = df[df['Metric_Type'] == 'SEM'].copy()    # Create explicit copies
            
            # Extract severity levels in order and bar categories
            severities = df_mean['Severity'].tolist()
            categories = ['Hard Misses', 'Misses', 'Reached']
            
            # Convert raw counts to percentages for both mean and SEM
            for severity in severities:
                # Get the total number of targets for this severity
                total_targets = df_mean.loc[df_mean['Severity'] == severity, 'Total Targets'].values[0]
                
                # Calculate percentages for means and SEMs
                for category in categories:
                    # Calculate percentage for mean values
                    mean_value = df_mean.loc[df_mean['Severity'] == severity, category].values[0]
                    df_mean.loc[df_mean['Severity'] == severity, f"{category}_pct"] = (mean_value / total_targets * 100)
                    
                    # Calculate percentage for SEM values
                    sem_value = df_sem.loc[df_sem['Severity'] == severity, category].values[0]
                    df_sem.loc[df_sem['Severity'] == severity, f"{category}_pct"] = (sem_value / total_targets * 100)
            
            # Create a grouped bar chart using percentages
            for j, category in enumerate(categories):
                # Only include in legend if this is the first disturbance type
                showlegend = category not in legend_added
                
                # Get SEM values for error bars
                error_y = df_sem[f"{category}_pct"].tolist()
                
                fig.add_trace(
                    go.Bar(
                        x=severities,
                        y=df_mean[f"{category}_pct"],
                        name=category,
                        marker_color='#F44336' if category == 'Hard Misses' else 
                                   '#FFC107' if category == 'Misses' else '#4CAF50',
                        text=[f"{x:.1f}%" for x in df_mean[f"{category}_pct"]],
                        textposition='outside',  # Show text above bars
                        error_y=dict(
                            type='data',
                            array=error_y,
                            visible=True,
                            thickness=1.5,
                            width=4
                        ),
                        showlegend=showlegend,  # Only show in legend once
                    ),
                    row=row, col=1
                )
                
                # Mark this category as added to the legend
                legend_added[category] = True
            
            # Update layout for this subplot
            fig.update_xaxes(title_text="Severity Level", row=row, col=1)
            fig.update_yaxes(title_text="Percentage (%)", row=row, col=1)
        
        if not has_data:
            print(f"No data found for agent type: {agent_type}. Skipping plot generation.")
            continue
            
        # Update overall figure layout
        fig.update_layout(
            title_text=f"Success Metrics for {agent_type.capitalize()} Agent",
            barmode='group',
            height=300 * len(disturbances),
            width=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Apply the y-axis range to all y-axes, add padding for the text above bars
            yaxis1=dict(range=[0, 110]),  # First subplot
            yaxis2=dict(range=[0, 110]),  # Second subplot
            yaxis3=dict(range=[0, 110]),  # Third subplot
        )
        
        # Store the figure
        figures[agent_type] = fig
        
        # Save the figure if requested
        if save_plots:
            # Make sure output directory exists
            plots_dir = os.path.join(base_dir, "..", "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Save as HTML
            html_output = os.path.join(plots_dir, f"success_metrics_{agent_type}.html")
            fig.write_html(html_output)
            print(f"Plot saved to {html_output}")
            
            # Save as PDF
            try:
                pdf_output = os.path.join(plots_dir, f"success_metrics_{agent_type}.pdf")
                fig.write_image(pdf_output)
                print(f"Plot saved to {pdf_output}")
            except Exception as e:
                print(f"Could not save PDF: {e}")
                print("To save PDF files, install kaleido: pip install kaleido")
        
        # Show the figure if requested
        if show_plots:
            fig.show()
    
    return figures

def main():
    parser = argparse.ArgumentParser(
        description="Create bar charts showing success metrics for each agent type."
    )
    parser.add_argument(
        "--save", action="store_true", default=True, 
        help="Save plots as HTML files (default: True)"
    )
    parser.add_argument(
        "--show", action="store_true", 
        help="Try to show plots in browser (may not work on remote servers)"
    )
    
    args = parser.parse_args()
    plot_success_metrics(save_plots=args.save, show_plots=args.show)

if __name__ == "__main__":
    main()