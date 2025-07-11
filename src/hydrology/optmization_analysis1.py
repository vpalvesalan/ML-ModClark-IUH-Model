import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def simulate_storm_data(num_watersheds=50, events_per_watershed_range=(10, 50)):
    """
    Simulates optimized storm event data for multiple watersheds.
    This function creates a realistic dataset to test the analysis workflow.
    """
    print("Simulating storm event data...")
    data = []
    watershed_ids = [f'ws_{1000 + i}' for i in range(num_watersheds)]

    for ws_id in watershed_ids:
        # --- Simulate base parameters for the watershed ---
        # Some watersheds will be more "coherent" than others
        base_R = np.random.uniform(1, 10)
        base_Tc = np.random.uniform(5, 24)
        coherence_factor = np.random.uniform(0.1, 0.9) # Lower is more coherent

        num_events = np.random.randint(*events_per_watershed_range)

        for _ in range(num_events):
            # --- Simulate storm-specific variations ---
            R_variation = np.random.normal(0, base_R * coherence_factor)
            Tc_variation = np.random.normal(0, base_Tc * coherence_factor)

            opt_R = max(0.1, base_R + R_variation)
            opt_Tc = max(0.1, base_Tc + Tc_variation)

            # --- Simulate storm metadata ---
            total_ppt_mm = np.random.uniform(10, 200)
            ppt_duration_min = np.random.uniform(30, 1440)
            peak_qf = np.random.uniform(5, 500)
            spatial_var = np.random.uniform(0, 1)
            
            antecedent = {
                'antecedent_14d': np.random.uniform(0, 150),
                'antecedent_7d': np.random.uniform(0, 100),
                'antecedent_3d': np.random.uniform(0, 50),
                'antecedent_24h': np.random.uniform(0, 30),
            }

            # Add some correlation between antecedent moisture and R
            opt_R -= antecedent['antecedent_7d'] * 0.01 * np.random.random()
            opt_R = max(0.1, opt_R)

            event_data = {
                'watershed_id': ws_id,
                'R': opt_R,
                'Tc': opt_Tc,
                'total_precipitation_mm': total_ppt_mm,
                'ppt_dutation_min': ppt_duration_min,
                'peak_quickflow': peak_qf,
                'ppt_spatial_variability': spatial_var,
                **antecedent
            }
            data.append(event_data)
            
    df = pd.DataFrame(data)
    print("Simulation complete.")
    return df

def analyze_watershed_coherence(df, r_cv_threshold=0.5, tc_cv_threshold=0.5):
    """
    Analyzes the coherence of R and Tc parameters within each watershed.
    It flags watersheds where the parameters are too variable.
    """
    print("\n--- Step 1: Analyzing Watershed Coherence ---")
    
    # Calculate stats for each watershed
    watershed_stats = df.groupby('watershed_id').agg(
        R_mean=('R', 'mean'),
        R_std=('R', 'std'),
        Tc_mean=('Tc', 'mean'),
        Tc_std=('Tc', 'std'),
        event_count=('R', 'count')
    ).reset_index()

    # Calculate Coefficient of Variation (CV)
    watershed_stats['R_cv'] = watershed_stats['R_std'] / watershed_stats['R_mean']
    watershed_stats['Tc_cv'] = watershed_stats['Tc_std'] / watershed_stats['Tc_mean']

    # Flag incoherent watersheds
    watershed_stats['incoherent'] = (watershed_stats['R_cv'] > r_cv_threshold) | \
                                    (watershed_stats['Tc_cv'] > tc_cv_threshold)

    incoherent_ws = watershed_stats[watershed_stats['incoherent']]
    
    print(f"Found {len(incoherent_ws)} incoherent watersheds based on CV thresholds (R > {r_cv_threshold}, Tc > {tc_cv_threshold}).")
    if not incoherent_ws.empty:
        print("Incoherent Watersheds Summary:")
        print(incoherent_ws[['watershed_id', 'R_cv', 'Tc_cv', 'event_count']].round(2))
    
    # Plot distribution of CVs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(watershed_stats['R_cv'], ax=axes[0], kde=True, bins=20)
    axes[0].axvline(r_cv_threshold, color='r', linestyle='--', label=f'Threshold ({r_cv_threshold})')
    axes[0].set_title('Distribution of R Coefficient of Variation (CV)')
    axes[0].set_xlabel('R CV')
    axes[0].legend()

    sns.histplot(watershed_stats['Tc_cv'], ax=axes[1], kde=True, bins=20)
    axes[1].axvline(tc_cv_threshold, color='r', linestyle='--', label=f'Threshold ({tc_cv_threshold})')
    axes[1].set_title('Distribution of Tc Coefficient of Variation (CV)')
    axes[1].set_xlabel('Tc CV')
    axes[1].legend()
    
    plt.suptitle('Watershed Parameter Coherence Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    coherent_ws_ids = watershed_stats[~watershed_stats['incoherent']]['watershed_id']
    return df[df['watershed_id'].isin(coherent_ws_ids)], watershed_stats

def identify_storm_outliers(df, watershed_stats, z_score_threshold=2.5):
    """
    Identifies outlier storm events within coherent watersheds using Z-scores.
    """
    print(f"\n--- Step 2: Identifying Storm Event Outliers (Z-score > {z_score_threshold}) ---")
    
    # Merge stats back to the main dataframe
    df_merged = df.merge(watershed_stats[['watershed_id', 'R_mean', 'Tc_mean', 'R_std', 'Tc_std']], on='watershed_id')
    
    # Calculate Z-scores for each event
    df_merged['R_zscore'] = (df_merged['R'] - df_merged['R_mean']) / df_merged['R_std']
    df_merged['Tc_zscore'] = (df_merged['Tc'] - df_merged['Tc_mean']) / df_merged['Tc_std']

    # Flag outliers
    df_merged['is_outlier'] = (abs(df_merged['R_zscore']) > z_score_threshold) | \
                              (abs(df_merged['Tc_zscore']) > z_score_threshold)

    outlier_events = df_merged[df_merged['is_outlier']]
    
    print(f"Found {len(outlier_events)} outlier storm events across {outlier_events['watershed_id'].nunique()} watersheds.")
    
    if not outlier_events.empty:
        print("Example Outlier Events:")
        print(outlier_events[['watershed_id', 'R', 'R_mean', 'R_zscore', 'Tc', 'Tc_mean', 'Tc_zscore']].round(2).head())

        # Visualization of outliers for a sample watershed
        sample_ws_id = outlier_events['watershed_id'].iloc[0]
        sample_ws_data = df_merged[df_merged['watershed_id'] == sample_ws_id]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sample_ws_data, x='R', y='Tc', hue='is_outlier', palette={True: 'red', False: 'blue'}, s=100)
        plt.title(f'Outlier Visualization for Sample Watershed: {sample_ws_id}')
        plt.xlabel('Optimized R')
        plt.ylabel('Optimized Tc')
        plt.legend(title='Is Outlier?')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    clean_df = df_merged[~df_merged['is_outlier']].copy()
    return clean_df

def analyze_parameter_drivers(df, watershed_stats):
    """
    Analyzes correlations between storm characteristics and parameter deviations.
    """
    print("\n--- Step 3: Analyzing Drivers of Parameter Variation ---")
    
    # Calculate deviations from the mean for each storm
    df['R_dev'] = df['R'] - df['R_mean']
    df['Tc_dev'] = df['Tc'] - df['Tc_mean']

    metadata_cols = [
        'total_precipitation_mm', 'ppt_dutation_min', 'peak_quickflow',
        'ppt_spatial_variability', 'antecedent_14d', 'antecedent_7d',
        'antecedent_3d', 'antecedent_24h'
    ]
    
    correlation_cols = ['R_dev', 'Tc_dev'] + metadata_cols
    correlation_matrix = df[correlation_cols].corr()

    # Plotting the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix[['R_dev', 'Tc_dev']], 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Correlation Between Storm Metadata and Parameter Deviations')
    plt.show()

    # Plotting the most significant relationship
    # Let's assume 'antecedent_7d' has a notable correlation with 'R_dev'
    strongest_corr_var = correlation_matrix['R_dev'].drop(['R_dev', 'Tc_dev']).abs().idxmax()
    
    print(f"Investigating strongest correlation for R_dev: '{strongest_corr_var}'")
    
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df, 
        x=strongest_corr_var, 
        y='R_dev', 
        scatter_kws={'alpha':0.3},
        line_kws={'color': 'red'}
    )
    plt.title(f'Deviation of R vs. {strongest_corr_var}')
    plt.xlabel(strongest_corr_var)
    plt.ylabel('Deviation of R from Watershed Mean')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    # In a real scenario, you would load your data here, e.g.,
    # storm_data = pd.read_csv('optimized_storm_parameters.csv')
    storm_data = simulate_storm_data(num_watersheds=50, events_per_watershed_range=(20, 100))
    
    print("\nOriginal Data Shape:", storm_data.shape)
    print("Number of unique watersheds:", storm_data['watershed_id'].nunique())
    print("\nData Head:")
    print(storm_data.head())
    
    # --- Step 1: Filter entire watersheds based on parameter variability ---
    # This step identifies and removes watersheds that are too inconsistent to be reliable.
    coherent_df, watershed_stats = analyze_watershed_coherence(
        storm_data, 
        r_cv_threshold=0.4, 
        tc_cv_threshold=0.4
    )
    print(f"\nData shape after removing incoherent watersheds: {coherent_df.shape}")
    
    # --- Step 2: Filter outlier storm events within the remaining watersheds ---
    # This step cleans the data further by removing individual anomalous storm events.
    if not coherent_df.empty:
        final_clean_df = identify_storm_outliers(
            coherent_df, 
            watershed_stats, 
            z_score_threshold=2.5
        )
        print(f"\nFinal clean data shape after removing outlier storms: {final_clean_df.shape}")
    else:
        print("\nNo coherent watersheds found. Halting analysis.")
        final_clean_df = pd.DataFrame()

    # --- Step 3: Analyze what drives the (non-outlier) parameter variations ---
    # This step helps understand the physical reasons for the remaining, acceptable variability.
    if not final_clean_df.empty:
        analyze_parameter_drivers(final_clean_df, watershed_stats)
    
    # --- Final Summary ---
    print("\n--- Workflow Complete ---")
    print(f"Initial number of events: {len(storm_data)}")
    if not final_clean_df.empty:
        print(f"Final number of events after filtering: {len(final_clean_df)}")
        print(f"Removed {len(storm_data) - len(final_clean_df)} events in total.")
        print(f"Final number of watersheds: {final_clean_df['watershed_id'].nunique()}")
        
        # The 'final_clean_df' is the dataset you would carry forward to your ML model.
        # You would likely calculate the final mean R and Tc per watershed from this dataframe.
        final_watershed_params = final_clean_df.groupby('watershed_id')[['R', 'Tc']].mean().reset_index()
        print("\nFinal Averaged Parameters for ML Model (from clean data):")
        print(final_watershed_params.head())

