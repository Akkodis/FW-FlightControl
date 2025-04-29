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

def find_metrics_file(base_dir, agent_type=None, disturbance=None, agent=None):
    """Find metrics file with the new naming convention."""
    if agent == "pid":
        # For PID, which doesn't have agent types
        search_pattern = os.path.join(base_dir, "pid", f"{disturbance}_pid_*.csv")
    else:
        # For agents with types (like TDMPC2)
        search_pattern = os.path.join(base_dir, agent_type, f"{disturbance}_*_{agent_type}_*.csv")
    
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"Warning: No matching metrics file found with pattern {search_pattern}")
        return None
    
    # Return the first matching file (there should typically be only one)
    return matching_files[0]

def plot_success_metrics(base_dir=None, save_plots=True, show_plots=False, 
                         selected_agents=None, selected_agent_types=None, selected_disturbances=None):
    """
    Create bar charts showing Success, Misses, and Hard Misses as percentages for each agent type.
    
    Args:
        base_dir: Directory containing the metrics. If None, uses the script's directory.
        save_plots: Whether to save the plots as HTML files
        show_plots: Whether to display the plots in the browser
        selected_agents: List of agents to process (None for all)
        selected_agent_types: List of agent types to process (None for all)
        selected_disturbances: List of disturbances to process (None for all)
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we can show plots
    if show_plots and is_headless():
        print("Warning: Running in headless environment. Cannot show plots in browser.")
        print("Plots will only be saved as HTML files.")
        show_plots = False
        save_plots = True
    
    # Define default values
    agents = ["tdmpc2", "pid"]
    agent_types = ["planning", "noplan", "truemf"]
    disturbances = ["turb", "gusts", "wind"]
    
    # Filter based on user selection
    if selected_agents:
        agents = [agent for agent in agents if agent in selected_agents]
        print(f"Filtering for agents: {agents}")
    
    if selected_agent_types:
        agent_types = [agent_type for agent_type in agent_types if agent_type in selected_agent_types]
        print(f"Filtering for agent types: {agent_types}")
    
    if selected_disturbances:
        disturbances = [dist for dist in disturbances if dist in selected_disturbances]
        print(f"Filtering for disturbances: {disturbances}")
    
    severity_colors = {
        "off": "#4CAF50",        # Green
        "light": "#2196F3",      # Blue
        "moderate": "#FFC107",   # Yellow/Amber
        "severe": "#F44336"      # Red
    }
    
    # Dictionary to store all figures
    figures = {}
    
    # Process each agent
    for agent in agents:
        if agent == "tdmpc2":
            # For TDMPC2, process each agent type
            for agent_type in agent_types:
                process_agent_metrics(agent, agent_type, base_dir, disturbances, figures, save_plots, show_plots)
        else:
            # For PID, which doesn't have agent types
            process_agent_metrics(agent, None, base_dir, disturbances, figures, save_plots, show_plots)
    
    return figures

def process_agent_metrics(agent, agent_type, base_dir, disturbances, figures, save_plots, show_plots):
    """Process metrics for a specific agent (and agent_type if applicable)"""
    agent_label = f"{agent}_{agent_type}" if agent_type else agent
    print(f"\n***** Processing plots for {agent_label}...")
    
    # Check if agent type directory exists (only for agents with types)
    if agent_type:
        agent_dir = os.path.join(base_dir, '..', 'metrics', agent_type)
        if not os.path.exists(agent_dir):
            print(f"Warning: Directory {agent_dir} does not exist. Skipping.")
            return
    
    # Create a subplot figure for this agent
    fig = make_subplots(
        rows=len(disturbances), 
        cols=1,
        subplot_titles=[f"{dist.capitalize()} Disturbance" for dist in disturbances],
        vertical_spacing=0.15
    )
    
    has_data = False  # Flag to check if any data was found
    legend_added = {}  # Dictionary to track which categories have been added to the legend
    
    for i, disturbance in enumerate(disturbances):
        row = i + 1
        
        # Find metrics file with the new naming convention
        metrics_file = find_metrics_file(base_dir, agent_type, disturbance, agent)
        
        if not metrics_file or not os.path.exists(metrics_file):
            print(f"Warning: No metrics file found for {agent_label} with {disturbance}. Skipping.")
            continue
        
        print(f"Using metrics file: {os.path.basename(metrics_file)}")
        has_data = True
        
        # Read the CSV file
        df = pd.read_csv(metrics_file)
        
        # Handle PID and TDMPC2 differently since PID doesn't have Mean/SEM data
        if agent == "pid":
            # For PID, use the values directly (no Mean/SEM)
            df_for_plot = df.copy()
            has_sem = False
        else:
            # For TDMPC2, filter for Mean and SEM
            df_mean = df[df['Metric_Type'] == 'Mean'].copy()
            df_sem = df[df['Metric_Type'] == 'SEM'].copy()
            df_for_plot = df_mean
            has_sem = True
        
        # Extract severity levels in order and bar categories
        severities = df_for_plot['Severity'].tolist()
        categories = ['Hard Misses', 'Misses', 'Reached']
        
        # Convert raw counts to percentages
        for severity in severities:
            # Get the total number of targets for this severity
            total_targets = df_for_plot.loc[df_for_plot['Severity'] == severity, 'Total Targets'].values[0]
            
            # Calculate percentages
            for category in categories:
                mean_value = df_for_plot.loc[df_for_plot['Severity'] == severity, category].values[0]
                df_for_plot.loc[df_for_plot['Severity'] == severity, f"{category}_pct"] = (mean_value / total_targets * 100)
                
                # Calculate SEM percentages if available (only for TDMPC2)
                if has_sem:
                    sem_value = df_sem.loc[df_sem['Severity'] == severity, category].values[0]
                    df_sem.loc[df_sem['Severity'] == severity, f"{category}_pct"] = (sem_value / total_targets * 100)
        
        # Create a grouped bar chart using percentages
        for j, category in enumerate(categories):
            # Only include in legend if this is the first disturbance type
            showlegend = category not in legend_added
            
            # Prepare error bars if SEM data is available
            error_y = None
            if has_sem:
                error_y = dict(
                    type='data',
                    array=df_sem[f"{category}_pct"].tolist(),
                    visible=True,
                    thickness=1.5,
                    width=4
                )
            
            fig.add_trace(
                go.Bar(
                    x=severities,
                    y=df_for_plot[f"{category}_pct"],
                    name=category,
                    marker_color='#F44336' if category == 'Hard Misses' else 
                            '#FFC107' if category == 'Misses' else '#4CAF50',
                    text=[f"{x:.1f}%" for x in df_for_plot[f"{category}_pct"]],
                    textposition='outside',  # Show text above bars
                    error_y=error_y,  # Only include error bars if SEM data is available
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
        print(f"No data found for {agent_label}. Skipping plot generation.")
        return
        
    # Update overall figure layout
    title_text = f"Success Metrics for {agent_type.capitalize()} Agent" if agent_type else f"Success Metrics for PID Agent"
    fig.update_layout(
        title_text=title_text,
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
    key = agent_type if agent_type else agent
    figures[key] = fig
    
    # Save the figure if requested
    if save_plots:
        # Make sure output directory exists
        plots_dir = os.path.join(base_dir, "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save as HTML
        filename_base = f"success_metrics_{agent_type}" if agent_type else f"success_metrics_{agent}"
        html_output = os.path.join(plots_dir, f"{filename_base}.html")
        fig.write_html(html_output)
        print(f"Plot saved to {html_output}")
        
        # Save as PDF
        try:
            pdf_output = os.path.join(plots_dir, f"{filename_base}.pdf")
            fig.write_image(pdf_output)
            print(f"Plot saved to {pdf_output}")
        except Exception as e:
            print(f"Could not save PDF: {e}")
            print("To save PDF files, install kaleido: pip install kaleido")
    
    # Show the figure if requested
    if show_plots:
        fig.show()

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
    parser.add_argument(
        "--agent", nargs="+", choices=["tdmpc2", "pid"],
        help="Specify which agent(s) to plot (default: all)"
    )
    parser.add_argument(
        "--agent-type", nargs="+", choices=["planning", "noplan", "truemf"],
        help="Specify which agent type(s) to plot (default: all)"
    )
    parser.add_argument(
        "--disturbance", nargs="+", choices=["turb", "gusts", "wind"],
        help="Specify which disturbance(s) to plot (default: all)"
    )
    
    args = parser.parse_args()
    plot_success_metrics(
        save_plots=args.save, 
        show_plots=args.show,
        selected_agents=args.agent,
        selected_agent_types=args.agent_type,
        selected_disturbances=args.disturbance
    )

if __name__ == "__main__":
    main()