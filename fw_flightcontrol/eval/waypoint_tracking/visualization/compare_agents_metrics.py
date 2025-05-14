import pandas as pd
import plotly.graph_objects as go
import os
import argparse
import plotly.io as pio

pio.kaleido.scope.mathjax = None

def plot_barcharts(base_dir, dist_type:str=None):
    # Load the CSV
    csv_path = os.path.join(base_dir, f"{dist_type}_pid_vs_tdmpc.csv")  # Update with your file path
    df = pd.read_csv(csv_path)

    # Clean up: normalize column names
    df.columns = df.columns.str.strip()

    # Filter PID rows and TD-MPC "Mean" + "SEM" rows
    pid_df = df[df["Agent"] == "PID"]
    tdmpc_mean_df = df[(df["Agent"] == "TD-MPC") & (df["Metric_Type"] == "Mean")]
    tdmpc_sem_df = df[(df["Agent"] == "TD-MPC") & (df["Metric_Type"] == "SEM")]

    # Merge TD-MPC mean and SEM rows
    tdmpc_df = tdmpc_mean_df.copy()
    tdmpc_df["SEM"] = tdmpc_sem_df["Reached"].values

    # Combine PID and TD-MPC into one dataframe
    pid_df["SEM"] = None
    combined_df = pd.concat([pid_df, tdmpc_df], ignore_index=True)

    # Ensure severity order
    severity_order = ["light", "moderate", "severe"]
    combined_df["Severity"] = pd.Categorical(combined_df["Severity"], categories=severity_order, ordered=True)

    # Plotting
    fig = go.Figure()

    for agent in combined_df["Agent"].unique():
        agent_data = combined_df[combined_df["Agent"] == agent]
        fig.add_trace(go.Bar(
            x=agent_data["Severity"],
            y=agent_data["Reached"],
            name=agent,
            text=[f'({s:.0f}%)' for s in agent_data["Success Rate (%)"]],
            textfont=dict(size=20),
            textposition='outside',
            error_y=dict(
                type='data',
                array=agent_data["SEM"].fillna(0),
                visible=True
            )
        ))

    # Layout
    fig.update_layout(
        title=f"Waypoints Reached by Agent and {dist_type.capitalize()} Severity",
        xaxis_title="Severity",
        yaxis_title="Waypoints Reached (%)",
        barmode='group',
        template='plotly_white',
        font=dict(size=20),
        # Add padding to the top of the chart
        yaxis=dict(
            rangemode='tozero',
            automargin=True,
            # Add 15% padding to the top to ensure text fits
            range=[0, combined_df["Reached"].max() * 1.15]
        ),
        margin=dict(t=100)  # Increase top margin
    )

    # Make sure output directory exists
    plots_dir = os.path.join(base_dir, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
        
    # Save as HTML
    filename_base = f"pid_vs_tdmpc_{dist_type}_barchart"
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


def main():
    parser = argparse.ArgumentParser(description="Plot bar charts from CSV data.")
    parser.add_argument(
        "--disturbance", type=str, choices=["turb", "gusts", "wind"],
        help="Specify which disturbance(s) to plot (default: all)"
    )
    
    args = parser.parse_args()
    plot_barcharts(base_dir='outputs/metrics', dist_type=args.disturbance)


if __name__ == "__main__":
    main()