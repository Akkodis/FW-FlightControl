import os
import pandas as pd
import numpy as np
import argparse
import re
import glob

def extract_filename_components(filename, agent_type):
    """Extract agent and checkpoint base from filename."""
    parts = filename.split('_')
    
    # First part is the disturbance type (turb/gusts)
    # Second part is usually the agent name (e.g., tdmpc2)
    agent = parts[1]
    
    # Extract the checkpoint base - different pattern for different agent types
    if agent_type == "planning":
        # For planning, format is usually: disturbance_agent_wp_checkpoint_seed.csv
        checkpoint_base = "_".join(parts[2:-1])
    else:
        # For others, format is: disturbance_agent_wp_agenttype_checkpoint_seed.csv
        checkpoint_base = "_".join(parts[3:-1])
    
    return agent, checkpoint_base

def synthesize_metrics(agent_type):
    """
    Compute mean metrics across all seeds for a given agent type.
    
    Args:
        agent_type (str): The agent type (noplan, planning, truemf)
    """
    metrics_dir_path = 'outputs/metrics'
    disturbances = ["noatmo", "wind", "turb", "gusts"]
    seeds = ["1", "2164", "2989", "4508"]
    
    # Dictionary to store results for each disturbance
    results = {}
    
    for disturbance in disturbances:
        print(f"Processing {disturbance} data for {agent_type}...")
        
        # Path to the directory containing CSV files for this agent and disturbance
        csv_dir = os.path.join(metrics_dir_path, agent_type, disturbance)
        
        if not os.path.exists(csv_dir):
            print(f"Warning: Directory {csv_dir} does not exist. Skipping.")
            continue
        
        # Find all CSV files for the given seeds using glob pattern
        csv_files = []
        for seed in seeds:
            # Use a more general pattern that will match various naming conventions
            pattern = f"{disturbance}_*_{seed}.csv"
            matching_files = glob.glob(os.path.join(csv_dir, pattern))
            csv_files.extend(matching_files)
        
        if not csv_files:
            print(f"No data found for {agent_type} with {disturbance}. Skipping.")
            continue
        
        # Extract agent and checkpoint base from the first CSV filename
        first_filename = os.path.basename(csv_files[0])
        agent, checkpoint_base = extract_filename_components(first_filename, agent_type)
        print(f"Detected agent: {agent}, checkpoint base: {checkpoint_base}")
        
        # Read all found CSV files
        all_dfs = []
        for csv_path in csv_files:
            try:
                # Extract seed from filename
                filename = os.path.basename(csv_path)
                seed_match = re.search(r'_(\d+)\.csv$', filename)
                if seed_match:
                    seed = seed_match.group(1)
                else:
                    seed = "unknown"
                    
                df = pd.read_csv(csv_path)
                df['Seed'] = seed
                all_dfs.append(df)
                print(f"Loaded: {csv_path}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        
        if not all_dfs:
            print(f"No valid data found for {agent_type} with {disturbance}. Skipping.")
            continue
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Columns for which we'll calculate SEM (exclude 'Total Targets' and non-numeric columns)
        sem_cols = [col for col in combined_df.columns if col not in ['Seed', 'Severity', 'Total Targets']]
        
        # Create a DataFrame to store results with interleaved means and SEMs
        result_metrics = pd.DataFrame()
        
        # Calculate mean and SEM for each severity level
        for severity in combined_df['Severity'].unique():
            severity_df = combined_df[combined_df['Severity'] == severity]
            
            # Calculate mean row
            mean_row = pd.DataFrame({'Severity': [severity], 'Metric_Type': ['Mean']})
            for col in combined_df.columns:
                if col not in ['Seed', 'Severity', 'Metric_Type']:
                    mean_row[col] = severity_df[col].mean()
            
            # Calculate SEM row
            sem_row = pd.DataFrame({'Severity': [severity], 'Metric_Type': ['SEM']})
            for col in sem_cols:
                sem_row[col] = severity_df[col].sem()
            
            # For columns where SEM is not calculated, use NaN
            for col in [c for c in combined_df.columns if c not in sem_cols and c not in ['Seed', 'Severity', 'Metric_Type']]:
                sem_row[col] = np.nan
            
            # Add rows to the result DataFrame
            result_metrics = pd.concat([result_metrics, mean_row, sem_row], ignore_index=True)
        
        # Add a column indicating the number of seeds averaged
        result_metrics['Seeds_Averaged'] = len(all_dfs)
        
        # Store the results
        results[disturbance] = result_metrics
        
        # Make sure the agent directory exists
        agent_dir = os.path.join(metrics_dir_path, agent_type)
        os.makedirs(agent_dir, exist_ok=True)
        
        # Save the results to a CSV file with the new naming convention
        output_file = os.path.join(agent_dir, f"{disturbance}_{agent}_{agent_type}_{checkpoint_base}.csv")
        result_metrics.to_csv(output_file, index=False)
        print(f"Mean metrics for {agent_type} with {disturbance} saved to {output_file}")
        
        # Print a summary of the results with SEM
        print("\nSummary of mean metrics with standard error:")
        # Get only the mean rows for summary calculations
        mean_rows = result_metrics[result_metrics['Metric_Type'] == 'Mean']
        sem_rows = result_metrics[result_metrics['Metric_Type'] == 'SEM']
        
        success_mean = mean_rows['Success Rate (%)'].mean()
        success_sem = sem_rows['Success Rate (%)'].mean() 
        print(f"Success Rate: {success_mean:.2f}% ± {success_sem:.2f}%")
        
        distance_mean = mean_rows['Avg Distance (m)'].mean()
        distance_sem = sem_rows['Avg Distance (m)'].mean()
        print(f"Avg Distance: {distance_mean:.2f} ± {distance_sem:.2f} m")
        
        time_mean = mean_rows['Avg Time (s)'].mean()
        time_sem = sem_rows['Avg Time (s)'].mean()
        print(f"Avg Time: {time_mean:.2f} ± {time_sem:.2f} s")
        print("-" * 50)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Compute mean metrics across all seeds for a given agent type."
    )
    parser.add_argument(
        "--agent-type", 
        choices=["noplan", "planning", "truemf"],
        help="The agent type to process"
    )
    
    args = parser.parse_args()
    synthesize_metrics(args.agent_type)

if __name__ == "__main__":
    main()