import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_barcharts(base_dir, dist_type: str = None):
    """
    Loads agent comparison data and plots a grouped bar chart of waypoints reached
    using Seaborn and Matplotlib.
    """
    # --- 1. Data Loading and Preparation (same as original) ---
    csv_path = os.path.join(base_dir, f"{dist_type}_pid_vs_tdmpc.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Get PID data
    pid_df = df[df["Agent"] == "PID"].copy()
    pid_df["SEM"] = 0  # Set SEM to 0 for PID as it has no variance across seeds
    
    # Get all TDMPC variants
    tdmpc_agents = ["TD-MPC-SS-ActPos", "TD-MPC-PI-Throttle", "TD-MPC-SS-Vanilla"]
    
    # Process each TDMPC variant
    tdmpc_dfs = []
    for agent in tdmpc_agents:
        mean_df = df[(df["Agent"] == agent) & (df["Metric_Type"] == "Mean")].copy()
        sem_df = df[(df["Agent"] == agent) & (df["Metric_Type"] == "SEM")]
        
        if not mean_df.empty and not sem_df.empty:
            # Add SEM values to the mean dataframe
            mean_df["SEM"] = sem_df["Reached"].values
            tdmpc_dfs.append(mean_df)
    
    # Combine all dataframes
    all_dfs = [pid_df] + tdmpc_dfs
    combined_df = pd.concat(all_dfs, ignore_index=True)

    severity_order = ["light", "moderate", "severe"]
    combined_df["Severity"] = pd.Categorical(combined_df["Severity"], categories=severity_order, ordered=True)
    combined_df.sort_values("Severity", inplace=True)

    # --- 2. Plotting with Seaborn and Matplotlib ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 12))  # Even larger size to accommodate much larger fonts

    # Create the bar plot
    sns.barplot(
        data=combined_df,
        x="Severity",
        y="Reached",
        hue="Agent",
        ax=ax,
        palette="Set2",  # Changed palette to accommodate more agents
        edgecolor="black",
        linewidth=2.5,  # Thicker bar edges for better visibility
        err_kws={"linewidth":3},  # Even thicker error bars
        capsize=0.15  # Larger cap size for error bars
    )

    # Create a shallow COPY of the containers list before iterating
    bar_containers = ax.containers[:]

    # Add text annotations (Success Rate %) and error bars
    # Get unique agents in the order they appear in the plot
    unique_agents = combined_df["Agent"].unique()
    
    # Iterate through each container (one for each agent)
    for i, container in enumerate(bar_containers):
        # Get the agent name for the current container
        agent = unique_agents[i]
        
        # Iterate through each bar in the container
        for j, bar in enumerate(container):
            # The bar's index in the container corresponds to the severity index
            severity = severity_order[j]
            height = bar.get_height()
            x_center = bar.get_x() + bar.get_width() / 2.

            # Find the corresponding data row
            row = combined_df[(combined_df['Severity'] == severity) & (combined_df['Agent'] == agent)]
            if not row.empty:
                success_rate = row["Success Rate (%)"].iloc[0]
                sem = row["SEM"].iloc[0]
                
                # Default text position is just above the bar
                text_y_position = height
                
                # Add error bar only for the TDMPC agents (not PID)
                if agent != "PID":
                    ax.errorbar(x=x_center, y=height, yerr=sem, fmt='none', c='black', capsize=5)
                    # If there's an error bar, place text above it
                    text_y_position += sem
                
                # Add success rate text with a small offset
                ax.text(x_center, text_y_position + 0.5, f'({success_rate:.0f}%)', ha='center', va='bottom', fontsize=30)  # Even larger font size for paper

    # --- 3. Layout and Styling ---
    ax.set_title(f"Waypoints Reached by Agent and {dist_type.capitalize()} Severity", fontsize=36, pad=25)  # Even larger font size for paper
    ax.set_xlabel("Severity", fontsize=36)  # Even larger font size for paper
    ax.set_ylabel("Waypoints Reached", fontsize=36)  # Even larger font size for paper
    ax.tick_params(axis='both', which='major', labelsize=32)  # Even larger font size for paper
    
    # Adjust y-axis to make space for text
    y_max = combined_df["Reached"].max() + combined_df["SEM"].max()
    ax.set_ylim(0, y_max * 1.25)  # Increased space for text
    
    # Customize legend with shorter agent names for better readability
    handles, labels = ax.get_legend_handles_labels()
    # Create shorter labels for the legend
    short_labels = []
    for label in labels:
        if label == "PID":
            short_labels.append("PID")
        elif label == "TD-MPC-SS-Vanilla":
            short_labels.append("TDMPC-SS-Vanilla")
        elif label == "TD-MPC-SS-ActPos":
            short_labels.append("TDMPC-SS-ActPos")
        elif label == "TD-MPC-PI-Throttle":
            short_labels.append("TDMPC-PI-Throttle")
        else:
            short_labels.append(label)
    
    ax.legend(handles, short_labels, fontsize=35, frameon=True, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)  # Display legend in two rows
    plt.tight_layout()

    # --- 4. Saving the Plot ---
    plots_dir = os.path.join(base_dir, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
        
    filename_base = f"pid_vs_tdmpc_{dist_type}_barchart"
    
    # Save as PNG (standard for matplotlib)
    # png_output = os.path.join(plots_dir, f"{filename_base}.png")
    # fig.savefig(png_output, dpi=300)
    # print(f"Plot saved to {png_output}")
        
    # Save as PDF
    pdf_output = os.path.join(plots_dir, f"{filename_base}.pdf")
    fig.savefig(pdf_output)
    print(f"Plot saved to {pdf_output}")
        
    # Save as EPS
    eps_output = os.path.join(plots_dir, f"{filename_base}.eps")
    fig.savefig(eps_output, format='eps')
    print(f"Plot saved to {eps_output}")
    
    plt.close(fig) # Close the figure to free memory

def main():
    parser = argparse.ArgumentParser(description="Plot bar charts from CSV data.")
    parser.add_argument(
        "--disturbance", type=str, choices=["turb", "gusts", "wind"],
        help="Specify which disturbance(s) to plot (default: all)"
    )
    parser.add_argument(
        "--base_dir", type=str, default="outputs/manuscript/final",
        help="Base directory containing the CSV files (default: outputs/manuscript/final)"
    )
    
    args = parser.parse_args()
    plot_barcharts(base_dir=args.base_dir, dist_type=args.disturbance)


if __name__ == "__main__":
    main()