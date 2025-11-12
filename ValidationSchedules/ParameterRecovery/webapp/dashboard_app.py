"""
Streamlit Dashboard for Parameter Recovery Results

A lightweight, interactive web dashboard for visualizing and analyzing
parameter recovery experiment results.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Add path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ValidationSchedules.ParameterRecovery.webapp import utils


# Configure page
st.set_page_config(
    page_title="Parameter Recovery Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main dashboard application."""
    
    # Add title and description
    st.title("🔬 Parameter Recovery Dashboard")
    st.markdown("""
        This dashboard displays results from parameter recovery experiments.
        Select an experiment and runs to visualize the results.
    """)
    
    # Sidebar - Input for primary_storage_dir
    st.sidebar.header("Experiment Configuration")
    # Default path - user can modify
    default_storage_dir = str(Path(__file__).parent.parent.parent.parent / "Experimentations" / "ParameterRecovery")
    primary_storage_dir = st.sidebar.text_input(
        "Storage Directory",
        value=default_storage_dir,
        help="Path to Experimentations/ParameterRecovery directory"
    )
    
    # Check if storage directory exists
    if not os.path.exists(primary_storage_dir):
        st.error(f"Storage directory not found: {primary_storage_dir}")
        st.info("Please enter a valid path to the Experimentations/ParameterRecovery directory")
        st.stop()
    
    # List all available experiments in the directory
    try:
        all_items = os.listdir(primary_storage_dir)
        available_experiments = [
            item for item in all_items 
            if os.path.isdir(os.path.join(primary_storage_dir, item)) 
            and not item.startswith('.')
        ]
        available_experiments.sort()
    except Exception as e:
        st.error(f"Error reading experiments directory: {str(e)}")
        st.stop()
    
    if not available_experiments:
        st.warning(f"No experiments found in {primary_storage_dir}")
        st.info("Run parameter recovery experiments first to generate data.")
        st.stop()
    
    # Sidebar - Dropdown for experiment_id selection
    experiment_id = st.sidebar.selectbox(
        "Experiment ID",
        options=available_experiments,
        index=0,
        help="Select an experiment to analyze"
    )
    
    # Build experiment path
    experiment_path = os.path.join(primary_storage_dir, experiment_id)
    
    # List available runs
    available_runs = utils.list_available_runs(primary_storage_dir, experiment_id)
    if not available_runs:
        st.warning(f"No runs found in {experiment_path}")
        st.stop()
    
    # Sidebar - Checkboxes for run selection
    st.sidebar.header("Select Runs")
    st.sidebar.markdown(f"Found {len(available_runs)} runs")
    selected_runs = []
    for run_id in available_runs:
        run_dir = os.path.join(experiment_path, run_id)
        status = utils.check_run_status(run_dir)
        status_emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⏳"
        if st.sidebar.checkbox(f"{status_emoji} {run_id}", value=False):
            selected_runs.append(run_id)
    
    # Display message if no runs selected
    if not selected_runs:
        st.info("Please select at least one run from the sidebar")
        st.stop()
    
    # Create tabs for each selected run
    tabs = st.tabs(selected_runs)
    for tab, run_id in zip(tabs, selected_runs):
        with tab:
            display_run_results(experiment_path, run_id)


def display_run_results(experiment_path: str, run_id: str):
    """
    Display results for a single run in a tab.
    
    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory
    run_id : str
        Run identifier (e.g., 'run_001')
    """
    # Build run directory path
    run_dir = os.path.join(experiment_path, run_id)
    
    # Check run status
    status = utils.check_run_status(run_dir)
    if status == "FAILED":
        st.error(f"❌ Run {run_id} failed")
        # Try to read error file
        error_file = os.path.join(run_dir, 'ERROR')
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                st.code(f.read())
        return
    elif status == "IN_PROGRESS":
        st.warning(f"⏳ Run {run_id} is still in progress or incomplete")
        return
    
    # Load results
    try:
        results_df, extras_dict = utils.load_param_recov_results(run_dir)
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return
    
    # Create sub-tabs for different views
    subtab1, subtab2, subtab3 = st.tabs(["📊 Results Table", "📈 Figures", "📝 Fitting History"])
    
    # Sub-tab 1: Results Table
    with subtab1:
        st.header("Parameter Recovery Results")
        
        # Display main results DataFrame
        st.subheader("Parameter Comparison")
        st.dataframe(results_df, use_container_width=True)
        
        # Display negative log-likelihood table
        st.subheader("Negative Log-Likelihood Comparison")
        negll_df = utils.format_negll_table(extras_dict['negll'])
        st.dataframe(negll_df, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Run ID", extras_dict['run_id'])
            st.metric("Total Parameters", len(results_df))
        with col2:
            # Calculate mean absolute delta
            mean_abs_delta = results_df['Delta'].abs().mean()
            st.metric("Mean |Delta|", f"{mean_abs_delta:.4f}")
    
    # Sub-tab 2: Figures
    with subtab2:
        st.header("Visualization")
        
        # Load figure paths
        fig_paths = utils.load_param_recov_figs(run_dir, format='png')
        
        # Create checkboxes for figure selection
        st.subheader("Select Figures to Display")
        col1, col2 = st.columns(2)
        with col1:
            show_psychometric = st.checkbox("Psychometric Curve", value=True, key=f"{run_id}_psychometric")
            show_confidence = st.checkbox("Confidence Plot", value=True, key=f"{run_id}_confidence")
        with col2:
            show_link = st.checkbox("Link Function", value=True, key=f"{run_id}_link")
            show_dist = st.checkbox("Confidence Distribution", value=True, key=f"{run_id}_dist")
        
        # Display selected figures
        if show_psychometric and fig_paths['psychometric']:
            st.subheader("Psychometric Curve")
            st.image(fig_paths['psychometric'], use_column_width=True)
        
        if show_confidence and fig_paths['confidence']:
            st.subheader("Confidence vs Stimulus Intensity")
            st.image(fig_paths['confidence'], use_column_width=True)
        
        if show_link and fig_paths['link_function']:
            st.subheader("Confidence Link Function")
            st.image(fig_paths['link_function'], use_column_width=True)
        
        if show_dist and fig_paths['confidence_dist']:
            st.subheader("Confidence Distribution")
            st.image(fig_paths['confidence_dist'], use_column_width=True)
    
    # Sub-tab 3: Fitting History
    with subtab3:
        st.header("Fitting History / Log Output")
        
        # Load and display log content
        log_content = utils.load_param_recov_history(run_dir)
        
        # Display in a scrollable text area
        st.text_area(
            "Log Output",
            value=log_content,
            height=600,
            disabled=True
        )
        
        # Add download button for log
        st.download_button(
            label="Download Log File",
            data=log_content,
            file_name=f"{run_id}_log.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
