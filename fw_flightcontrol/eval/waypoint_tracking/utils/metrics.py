import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R


def compute_target_success(target_success, severity_range):
    """Compute and print target success statistics"""
    total_targets_array = np.zeros(len(severity_range))
    success_percent_array = np.zeros(len(severity_range))
    success_dict_array = []  # Will store dict_successes for each severity level

    print("\nTarget success statistics by severity level:")
    for sev_cnt, severity in enumerate(severity_range):
        unique, counts = np.unique(target_success[sev_cnt], return_counts=True)
        dict_successes = {'hard_miss': 0, 'missed': 0, 'reached': 0}
        for (u, c) in zip(unique, counts):
            if u == 0:
                dict_successes['hard_miss'] = int(c)
            elif u == 1:
                dict_successes['missed'] = int(c)
            elif u == 2:
                dict_successes['reached'] = int(c)
        
        success_dict_array.append(dict_successes)
        total_targets_array[sev_cnt] = sum(dict_successes.values())
        success_percent_array[sev_cnt] = (dict_successes.get('reached', 0) / total_targets_array[sev_cnt] * 100) if total_targets_array[sev_cnt] > 0 else 0

        print(f"  {severity} severity: {dict_successes} - Success rate: {success_percent_array[sev_cnt]:.1f}%")

    print(f"\nTotal targets array: {total_targets_array}")
    print(f"Success percentage array: {success_percent_array}")
    print(f"Success dictionaries by severity: {success_dict_array}")
    
    return total_targets_array, success_percent_array, success_dict_array

def compute_fcs_fluctuation(ep_fcs_fluct, severity_range):
    """Compute and print FCS fluctuation statistics"""
    avg_fcs_fluct_per_severity = np.zeros((len(severity_range), 3))
    
    print("\nFCS fluctuation statistics by severity level:")
    for sev_cnt, severity in enumerate(severity_range):
        # Average across all episodes for this severity level
        avg_fcs_fluct_per_severity[sev_cnt] = np.nanmean(ep_fcs_fluct[sev_cnt], axis=0)
        print(f"  {severity} severity: aileron={avg_fcs_fluct_per_severity[sev_cnt, 0]:.4f}, "
            f"elevator={avg_fcs_fluct_per_severity[sev_cnt, 1]:.4f}, "
            f"throttle={avg_fcs_fluct_per_severity[sev_cnt, 2]:.4f}")

    print(f"\nAverage FCS fluctuation array by severity: {avg_fcs_fluct_per_severity}")
    
    return avg_fcs_fluct_per_severity

def compute_distance(enu_positions, severity_range, targets_enu, num_ep):
    """Compute and print distance traveled statistics"""
    episode_distances = np.full((len(severity_range), num_ep), np.nan)
    episode_distances_normalized = np.full((len(severity_range), num_ep), np.nan)
    avg_distance_per_severity = np.zeros(len(severity_range))
    avg_distance_normalized_per_severity = np.zeros(len(severity_range))

    print("\nDistance traveled statistics by severity level:")
    for sev_cnt, severity in enumerate(severity_range):
        for ep_cnt in range(num_ep):
            # Get positions for this episode
            positions = enu_positions[sev_cnt, ep_cnt]
            
            # Find where valid positions end (first NaN)
            valid_idx = np.argmax(np.isnan(positions[:, 0]))
            if valid_idx == 0:  # No NaNs found
                valid_idx = len(positions)
            
            # Get only valid positions
            valid_positions = positions[:valid_idx]
            
            # Calculate distances between consecutive points
            if len(valid_positions) > 1:
                diffs = np.diff(valid_positions, axis=0)
                distances = np.sqrt(np.sum(diffs**2, axis=1))
                total_distance = np.sum(distances)
                distance_to_target = np.linalg.norm([0, 0, 600] - targets_enu[ep_cnt])
                episode_distances[sev_cnt, ep_cnt] = total_distance
                episode_distances_normalized[sev_cnt, ep_cnt] = total_distance / distance_to_target

        # Calculate average distance for this severity
        avg_distance = np.nanmean(episode_distances[sev_cnt])
        avg_distance_per_severity[sev_cnt] = avg_distance

        avg_distance_normalized = np.nanmean(episode_distances_normalized[sev_cnt])
        avg_distance_normalized_per_severity[sev_cnt] = avg_distance_normalized
        
        print(f"  {severity} severity: Mean distance = {avg_distance:.2f} m, Normalized = {avg_distance_normalized:.2f} m")

    print(f"\nAverage distance traveled by severity: {avg_distance_per_severity}, Normalized: {avg_distance_normalized_per_severity}")
    
    return episode_distances, avg_distance_per_severity, avg_distance_normalized_per_severity

def compute_time(enu_positions, severity_range, num_ep, fdm_dt):
    """Compute and print elapsed time statistics"""
    episode_times = np.full((len(severity_range), num_ep), np.nan)
    avg_time_per_severity = np.zeros(len(severity_range))

    print("\nTime elapsed statistics by severity level:")
    for sev_cnt, severity in enumerate(severity_range):
        for ep_cnt in range(num_ep):
            # Get positions for this episode
            positions = enu_positions[sev_cnt, ep_cnt]
            
            # Find where valid positions end (first NaN)
            valid_idx = np.argmax(np.isnan(positions[:, 0]))
            if valid_idx == 0:  # No NaNs found
                valid_idx = len(positions)
            
            # Calculate time elapsed (steps * dt)
            elapsed_time = valid_idx * fdm_dt
            episode_times[sev_cnt, ep_cnt] = elapsed_time
        
        # Calculate average time for this severity
        avg_time = np.nanmean(episode_times[sev_cnt])
        avg_time_per_severity[sev_cnt] = avg_time
        
        print(f"  {severity} severity: Mean time = {avg_time:.2f} s")
        for ep_cnt in range(num_ep):
            minutes = int(episode_times[sev_cnt, ep_cnt] / 60)
            seconds = episode_times[sev_cnt, ep_cnt] % 60
            # print(f"    Episode {ep_cnt}: {minutes}m {seconds:.2f}s ({episode_times[sev_cnt, ep_cnt]:.2f} s)")

    print(f"\nAverage time elapsed by severity: {avg_time_per_severity}")
    
    return episode_times, avg_time_per_severity

def save_metrics_summary(csv_filename, severity_range, total_targets_array, success_dict_array, success_percent_array, 
                         avg_fcs_fluct_per_severity, avg_distance_per_severity, avg_distance_normalized_per_severity, avg_time_per_severity):
    """Save all metrics to a CSV file"""

    if not os.path.exists("eval/waypoint_tracking/outputs"):
        os.makedirs("eval/waypoint_tracking/outputs")

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow([
            'Severity', 
            'Total Targets', 
            'Hard Misses', 'Misses', 'Reached',
            'Success Rate (%)',
            'Avg Aileron Fluctuation', 
            'Avg Elevator Fluctuation', 
            'Avg Throttle Fluctuation',
            'Avg Distance (m)',
            'Avg Distance Normalized (m)',
            'Avg Time (s)'
        ])
        
        # Write data for each severity level
        for sev_cnt, severity in enumerate(severity_range):
            dict_successes = success_dict_array[sev_cnt]
            
            writer.writerow([
                severity,
                int(total_targets_array[sev_cnt]),
                dict_successes['hard_miss'],
                dict_successes['missed'],
                dict_successes['reached'],
                f"{success_percent_array[sev_cnt]:.2f}",
                f"{avg_fcs_fluct_per_severity[sev_cnt, 0]:.6f}",
                f"{avg_fcs_fluct_per_severity[sev_cnt, 1]:.6f}",
                f"{avg_fcs_fluct_per_severity[sev_cnt, 2]:.6f}",
                f"{avg_distance_per_severity[sev_cnt]:.2f}",
                f"{avg_distance_normalized_per_severity[sev_cnt]:.2f}",
                f"{avg_time_per_severity[sev_cnt]:.2f}"
            ])
        
        # Add an overall average row
        avg_success_rate = np.mean(success_percent_array)
        writer.writerow([
            'AVERAGE',
            int(np.sum(total_targets_array)),
            sum(d['hard_miss'] for d in success_dict_array),
            sum(d['missed'] for d in success_dict_array),
            sum(d['reached'] for d in success_dict_array),
            f"{avg_success_rate:.2f}",
            f"{np.mean(avg_fcs_fluct_per_severity[:, 0]):.6f}",
            f"{np.mean(avg_fcs_fluct_per_severity[:, 1]):.6f}",
            f"{np.mean(avg_fcs_fluct_per_severity[:, 2]):.6f}",
            f"{np.mean(avg_distance_per_severity):.2f}",
            f"{np.mean(avg_distance_normalized_per_severity):.2f}",
            f"{np.mean(avg_time_per_severity):.2f}"
        ])

    print(f"Metrics summary saved to {csv_filename}")
    return csv_filename

def plot_trajectories(enu_positions, orientations, wind_vector, targets_enu, target_success, severity_range, dubins_paths, plot_frames=False):
    """Plot 3D trajectories for all severity levels"""
    R_ned2enu = np.array([
        [0, 1, 0],   # East  <- North
        [1, 0, 0],   # North <- East
        [0, 0, -1]   # Up    <- Down
    ])
    fig, axs = plt.subplots(1, len(severity_range), figsize=(15, 5), subplot_kw={'projection': '3d'}, squeeze=False)
    num_ep = targets_enu.shape[0]
    
    
    for sev_cnt, severity in enumerate(severity_range):
        ax = axs[0, sev_cnt]
        ax.scatter(enu_positions[sev_cnt, :, 0, 0], enu_positions[sev_cnt, :, 0, 1], enu_positions[sev_cnt, :, 0, 2], color='black', label='Start')
        
        # Track which success levels we've already added to the legend
        added_to_legend = set()
        
        # Plot targets with different colors based on success level
        for ep, target_success_ep in enumerate(target_success[sev_cnt]):
            target_color = 'red'    # Hard miss (0)
            label = 'Hard Missed'
            if target_success_ep == 1:
                target_color = 'orange'  # Missed (1)
                label = 'Missed'
            elif target_success_ep == 2:
                target_color = 'green'   # Reached (2)
                label = 'Reached'
            
            # Only add the label if we haven't seen this success level before
            if target_success_ep not in added_to_legend:
                ax.scatter(targets_enu[ep, 0], targets_enu[ep, 1], targets_enu[ep, 2], color=target_color, label=label, s=50)
                added_to_legend.add(target_success_ep)
            else:
                ax.scatter(targets_enu[ep, 0], targets_enu[ep, 1], targets_enu[ep, 2], color=target_color, s=50)

        # Calculate wind speed once per severity level (using the first episode's data)
        curr_sev_windvector = wind_vector[sev_cnt, 0, 1, :]
        wind_speed = np.linalg.norm(curr_sev_windvector)
        start_pos = enu_positions[sev_cnt, 0, 0, :]
        
        # Add wind text label once per severity level
        ax.text(
            start_pos[0], start_pos[1], start_pos[2] + 5,
            f"Wind: {wind_speed * 3.6:.1f} kph",
            color='black', fontsize=14,
        )

        # Plot trajectories
        for i in range(num_ep):
            ax.plot(enu_positions[sev_cnt, i, :, 0], enu_positions[sev_cnt, i, :, 1], enu_positions[sev_cnt, i, :, 2])
            if isinstance(dubins_paths[sev_cnt, i], np.ndarray):
                ax.scatter(dubins_paths[sev_cnt, i][:, 0], dubins_paths[sev_cnt, i][:, 1], dubins_paths[sev_cnt, i][:, 2], color='purple', s=30, label='Dubins Path')

            # Plot the wind vector at the first timestep
            # Convert wind vector from NED to ENU
            curr_sev_windvector = wind_vector[sev_cnt, i, 1, :]
            # wind_vector_enu = R_ned2enu @ curr_sev_windvector
            wind_vector_enu = [1, 1, -1] * curr_sev_windvector # invert down vector to up
    
            # Calculate a reasonable scale factor for the wind vector
            wind_speed = np.linalg.norm(curr_sev_windvector)
            scale_factor = 10.0  # Adjust this value to make the arrow an appropriate size
        
            # Place the wind vector at a visible location in the plot
            # Using the target position
            start_pos = enu_positions[sev_cnt, 0, 0, :]
        
            # Plot wind vector as an arrow
            # ax.quiver(
            #     targets_enu[i, 0], targets_enu[i, 1], targets_enu[i, 2],  # starting position
            #     wind_vector_enu[0], wind_vector_enu[1], wind_vector_enu[2],  # direction
            #     color='cyan', 
            #     linewidth=2,
            #     arrow_length_ratio=0.2,
            #     length=wind_speed
            # )
        

            
            # Plot frames if requested
            if plot_frames:
                for j in range(0, 2000, 50):
                    if np.isnan(enu_positions[sev_cnt, i, j, 0]):
                        break
                    x, y, z = enu_positions[sev_cnt, i, j]
                    roll, pitch, yaw = orientations[sev_cnt, i, j]
                    R_localned2body = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
                    R_enu2body = R_ned2enu @ R_localned2body

                    x_axis = R_enu2body[:, 0] * 5
                    y_axis = R_enu2body[:, 1] * 5
                    z_axis = -R_enu2body[:, 2] * 5
                    ax.quiver(x, y, z, *x_axis, color='red')
                    ax.quiver(x, y, z, *y_axis, color='green')
                    ax.quiver(x, y, z, *z_axis, color='blue')
        
        # Set axis properties
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(575, 625)
        ax.set_xlabel('E [m]', fontsize=17)
        ax.set_ylabel('N [m]', fontsize=17)
        ax.set_zlabel('U [m]', fontsize=17)
        ax.tick_params(labelsize=15)
        ax.legend(loc='upper left', fontsize=17)
        ax.set_title(f'TD-MPC 3D Trajectories - {severity}', fontsize=20)
        ax.grid()

    return fig


def plot_trajectories_plotly(enu_positions, orientations, wind_vector, targets_enu, target_success, severity_range, 
                             plot_frames=False, save_plot=True):
    """Plot 3D trajectories for all severity levels using Plotly in a 2x2 grid layout"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os
    import math
    
    # Define NED to ENU conversion matrix
    R_ned2enu = np.array([
        [0, 1, 0],   # East  <- North
        [1, 0, 0],   # North <- East
        [0, 0, -1]   # Up    <- Down
    ])
    
    # Define a color sequence to maintain consistency across severity levels
    # These are Plotly's default colors which ensure good visibility
    trajectory_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    
    # Calculate how many rows we need (2 columns per row)
    num_rows = math.ceil(len(severity_range) / 2)
    num_cols = min(len(severity_range), 2)  # Use 2 columns or fewer if not enough severity levels
    
    # Create subplot titles
    subplot_titles = [f'3D Trajectories - {severity}' for severity in severity_range]
    
    # Create subplots (multiple rows, 2 columns)
    specs = [[{'type': 'scene'} for _ in range(num_cols)] for _ in range(num_rows)]
    fig = make_subplots(
        rows=num_rows, 
        cols=num_cols,
        specs=specs,
        subplot_titles=subplot_titles
    )
    
    num_ep = targets_enu.shape[0]
    
    for sev_cnt, severity in enumerate(severity_range):
        # Calculate which row and column this severity level goes in
        row_idx = sev_cnt // 2 + 1  # 1-indexed
        col_idx = sev_cnt % 2 + 1   # 1-indexed
        
        # Plot starting positions
        fig.add_trace(
            go.Scatter3d(
                x=enu_positions[sev_cnt, :, 0, 0],
                y=enu_positions[sev_cnt, :, 0, 1],
                z=enu_positions[sev_cnt, :, 0, 2],
                mode='markers',
                marker=dict(size=5, color='black'),
                name='Start Positions',
                showlegend=(sev_cnt == 0)  # Only show in legend once
            ),
            row=row_idx, col=col_idx
        )
        
        # Plot targets with colors based on success level
        for ep, target_success_ep in enumerate(target_success[sev_cnt]):
            # Determine target color based on success level
            if target_success_ep == 0:
                target_color = 'red'      # Hard miss
                target_name = 'Hard Miss'
            elif target_success_ep == 1:
                target_color = 'orange'   # Missed
                target_name = 'Missed'
            else:  # target_success_ep == 2
                target_color = 'green'    # Reached
                target_name = 'Reached'
            
            # Group targets by success level to avoid legend duplication
            show_in_legend = False
            if sev_cnt == 0:
                if f"Target-{target_name}" not in [trace.name for trace in fig.data]:
                    show_in_legend = True
            
            # Add the target marker
            fig.add_trace(
                go.Scatter3d(
                    x=[targets_enu[ep, 0]],
                    y=[targets_enu[ep, 1]],
                    z=[targets_enu[ep, 2]],
                    mode='markers',
                    marker=dict(size=5, color=target_color, symbol='circle'),
                    name=f"Target-{target_name}",
                    showlegend=show_in_legend
                ),
                row=row_idx, col=col_idx
            )
            
            # Plot the wind vector at the target location
            curr_sev_windvector = wind_vector[sev_cnt, ep, 1, :]
            wind_vector_enu = [1, 1, -1] * curr_sev_windvector  # invert down vector to up
            
            # Calculate wind speed
            wind_speed = np.linalg.norm(curr_sev_windvector)
            
            # Use target position as starting point for wind vector
            target_pos = targets_enu[ep]
            
            # Create end position for the line
            scale_factor = 3.0  # Increased scale factor for better visibility
            end_pos = [
                target_pos[0] + (wind_vector_enu[0] * scale_factor),
                target_pos[1] + (wind_vector_enu[1] * scale_factor),
                target_pos[2] + (wind_vector_enu[2] * scale_factor)
            ]
            
            # 1. First draw the line for the wind vector
            fig.add_trace(
                go.Scatter3d(
                    x=[target_pos[0], end_pos[0]],
                    y=[target_pos[1], end_pos[1]],
                    z=[target_pos[2], end_pos[2]],
                    mode='lines',
                    line=dict(color='cyan', width=5),
                    name='Wind Vector',
                    showlegend=(sev_cnt == 0 and ep == 0)  # Only show once in legend
                ),
                row=row_idx, col=col_idx
            )
            
            # 2. Then add a cone at the end of the line
            fig.add_trace(
                go.Cone(
                    x=[end_pos[0]],
                    y=[end_pos[1]],
                    z=[end_pos[2]],
                    u=[wind_vector_enu[0]],
                    v=[wind_vector_enu[1]],
                    w=[wind_vector_enu[2]],
                    sizemode="absolute",
                    sizeref=3,  # Smaller than before for the arrowhead effect
                    colorscale=[[0, 'cyan'], [1, 'cyan']],
                    showscale=False,
                    showlegend=False  # Don't add this to legend since the line is already there
                ),
                row=row_idx, col=col_idx
            )
            
            # Add text label for wind speed and direction near target
            fig.add_trace(
                go.Scatter3d(
                    x=[target_pos[0]],
                    y=[target_pos[1]],
                    z=[target_pos[2] + 5],  # Position text slightly above target
                    mode='text',
                    text=[f"Wind: {wind_speed:.1f} mps"],
                    showlegend=False
                ),
                row=row_idx, col=col_idx
            )
        
        # Plot trajectory for each episode
        for i in range(num_ep):
            # Find where valid positions end (first NaN)
            valid_idx = np.argmax(np.isnan(enu_positions[sev_cnt, i, :, 0]))
            if valid_idx == 0:  # No NaNs found
                valid_idx = len(enu_positions[sev_cnt, i])
            
            # Extract valid positions only
            x_vals = enu_positions[sev_cnt, i, :valid_idx, 0]
            y_vals = enu_positions[sev_cnt, i, :valid_idx, 1]
            z_vals = enu_positions[sev_cnt, i, :valid_idx, 2]
            
            # Get consistent color for this trajectory based on episode index
            traj_color = trajectory_colors[i % len(trajectory_colors)]
            
            # Add trajectory line with consistent color
            fig.add_trace(
                go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals,
                    mode='lines',
                    line=dict(width=3, color=traj_color),
                    name=f'Episode {i}',
                    showlegend=(sev_cnt == 0 and i == 0)  # Only show one trajectory in legend
                ),
                row=row_idx, col=col_idx
            )
            
            
            # Plot frames if requested
            if plot_frames:
                for j in range(0, 2000, 50):
                    if np.isnan(enu_positions[sev_cnt, i, j, 0]):
                        break
                    x, y, z = enu_positions[sev_cnt, i, j]
                    roll, pitch, yaw = orientations[sev_cnt, i, j]
                    R_localned2body = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
                    R_enu2body = R_ned2enu @ R_localned2body

                    x_axis = R_enu2body[:, 0] * 3
                    y_axis = R_enu2body[:, 1] * 3
                    z_axis = -R_enu2body[:, 2] * 3
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x, x + x_axis[0]],
                            y=[y, y + x_axis[1]],
                            z=[z, z + x_axis[2]],
                            mode='lines',
                            line=dict(color='red', width=4),
                            showlegend=False
                        ),
                        row=row_idx, col=col_idx
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x, x + y_axis[0]],
                            y=[y, y + y_axis[1]],
                            z=[z, z + y_axis[2]],
                            mode='lines',
                            line=dict(color='green', width=4),
                            showlegend=False
                        ),
                        row=row_idx, col=col_idx
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x, x + z_axis[0]],
                            y=[y, y + z_axis[1]],
                            z=[z, z + z_axis[2]],
                            mode='lines',
                            line=dict(color='blue', width=4),
                            showlegend=False
                        ),
                        row=row_idx, col=col_idx
                    )

    # Configure the layout for all scenes
    layout_updates = {}
    for sev_cnt in range(len(severity_range)):
        scene_key = f'scene{sev_cnt+1}'
        layout_updates[scene_key] = dict(
            xaxis=dict(title='E', range=[-200, 200]),
            yaxis=dict(title='N', range=[-200, 200]),
            zaxis=dict(title='U', range=[565, 635]),
            aspectmode='cube'
        )
    
    # Set overall layout properties
    fig.update_layout(
        height=500 * num_rows,  # Adjusted height based on number of rows
        width=1000,  # Fixed width
        title=dict(
            text="Trajectory Visualization",
            x=0.5
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        **layout_updates  # Apply the scene configurations
    )
    
    # Save the plot as HTML if requested
    if save_plot:
        # Determine the output directory
        output_dir = os.path.join(os.getcwd(), "eval/waypoint_tracking/outputs/plots")
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the filename
        filename = os.path.join(output_dir, f"trajectories_plot.html")
        
        # Save the plot
        fig.write_html(
            filename,
            include_plotlyjs='cdn',
            full_html=True
        )
        print(f"Trajectory plot saved to {filename}")
    
    return fig
