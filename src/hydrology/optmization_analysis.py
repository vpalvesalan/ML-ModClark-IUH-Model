import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, linregress

def simulate_storm_data_with_trends(num_watersheds=50, events_per_watershed_range=(20, 100)):
    """
    Simulates storm data and corresponding land use data over a 10-year period.
    Introduces temporal trends in R and Tc for a subset of watersheds,
    linked to simulated land use change.
    """
    print("Simulating storm event and land use data...")
    storm_data = []
    land_use_data = []
    watershed_ids = [f'ws_{1000 + i}' for i in range(num_watersheds)]
    years = range(2014, 2024)

    # NLCD classes for simulation
    NLCD_CLASSES = {
        24: 'Developed, High Intensity',
        42: 'Evergreen Forest',
        82: 'Cultivated Crops'
    }

    for i, ws_id in enumerate(watershed_ids):
        # --- Simulate base parameters and land use ---
        base_R = np.random.uniform(1, 10)
        base_Tc = np.random.uniform(5, 24)
        coherence_factor = np.random.uniform(0.1, 0.5)

        # Simulate land use trend for a subset of watersheds
        dev_trend = 0
        forest_trend = 0
        if i < num_watersheds / 2: # Half the watersheds have a trend
            dev_trend = np.random.uniform(0.1, 0.8) # % increase per year
            forest_trend = -np.random.uniform(0.1, 0.8)

        initial_dev = np.random.uniform(1, 5)
        initial_forest = np.random.uniform(20, 50)
        
        # Correlate parameter trend with land use trend
        R_trend_factor = -dev_trend * 0.1 * np.random.random() # Urbanization can decrease R
        Tc_trend_factor = -dev_trend * 0.05 * np.random.random() # Urbanization can decrease Tc

        for year_idx, year in enumerate(years):
            # Simulate land use for the year
            land_use_data.append({
                'watershed_id': ws_id,
                'year': year,
                'developed_high_intensity_pct': max(0, initial_dev + dev_trend * year_idx),
                'evergreen_forest_pct': max(0, initial_forest + forest_trend * year_idx)
            })
            
            # Apply temporal trend to base parameters
            current_base_R = base_R + R_trend_factor * year_idx
            current_base_Tc = base_Tc + Tc_trend_factor * year_idx

            num_events = np.random.randint(events_per_watershed_range[0]/5, events_per_watershed_range[1]/5)
            if num_events == 0 and year_idx % 3 == 0: num_events = 2 # Ensure some events
            
            for _ in range(num_events):
                R_variation = np.random.normal(0, current_base_R * coherence_factor)
                Tc_variation = np.random.normal(0, current_base_Tc * coherence_factor)

                storm_data.append({
                    'watershed_id': ws_id,
                    'year': year,
                    'R': max(0.1, current_base_R + R_variation),
                    'Tc': max(0.1, current_base_Tc + Tc_variation),
                    'total_precipitation_mm': np.random.uniform(10, 200),
                    'antecedent_7d': np.random.uniform(0, 100)
                })

    storm_df = pd.DataFrame(storm_data)
    land_use_df = pd.DataFrame(land_use_data)
    print("Simulation complete.")
    return storm_df, land_use_df

def analyze_watershed_coherence(df, r_cv_threshold=0.5, tc_cv_threshold=0.5):
    """Analyzes the overall coherence of R and Tc within each watershed."""
    print("\n--- Step 1: Analyzing Overall Watershed Coherence ---")
    watershed_stats = df.groupby('watershed_id').agg(
        R_mean=('R', 'mean'), R_std=('R', 'std'),
        Tc_mean=('Tc', 'mean'), Tc_std=('Tc', 'std'),
        event_count=('R', 'count')
    ).reset_index()
    watershed_stats['R_cv'] = watershed_stats['R_std'] / watershed_stats['R_mean']
    watershed_stats['Tc_cv'] = watershed_stats['Tc_std'] / watershed_stats['Tc_mean']
    watershed_stats['incoherent'] = (watershed_stats['R_cv'] > r_cv_threshold) | (watershed_stats['Tc_cv'] > tc_cv_threshold)
    
    incoherent_ws_count = watershed_stats['incoherent'].sum()
    print(f"Found {incoherent_ws_count} incoherent watersheds based on overall CV thresholds.")
    
    coherent_ws_ids = watershed_stats[~watershed_stats['incoherent']]['watershed_id']
    return df[df['watershed_id'].isin(coherent_ws_ids)], watershed_stats

def analyze_temporal_and_land_use_trends(df, land_use_df, p_value_threshold=0.05):
    """
    Analyzes yearly trends in parameters and correlates them with land use change.
    """
    print(f"\n--- Step 2: Analyzing Temporal Trends (p-value < {p_value_threshold}) ---")
    trend_results = []
    
    for ws_id, group in df.groupby('watershed_id'):
        yearly_stats = group.groupby('year').agg(R_mean=('R', 'mean'), Tc_mean=('Tc', 'mean'), event_count=('R', 'count')).reset_index()
        
        # Need at least 3 years with data to detect a trend
        if len(yearly_stats) < 3:
            trend_results.append({'watershed_id': ws_id, 'has_temporal_trend': False, 'R_p_value': None, 'Tc_p_value': None})
            continue

        # Linear regression of parameter vs. year
        r_slope, _, _, r_p_value, _ = linregress(yearly_stats['year'], yearly_stats['R_mean'])
        tc_slope, _, _, tc_p_value, _ = linregress(yearly_stats['year'], yearly_stats['Tc_mean'])
        
        has_trend = (r_p_value < p_value_threshold) or (tc_p_value < p_value_threshold)
        trend_results.append({'watershed_id': ws_id, 'has_temporal_trend': has_trend, 'R_p_value': r_p_value, 'Tc_p_value': tc_p_value})

    trend_summary = pd.DataFrame(trend_results)
    df_with_trends = df.merge(trend_summary, on='watershed_id')
    
    trend_ws_count = trend_summary['has_temporal_trend'].sum()
    print(f"Found {trend_ws_count} watersheds with significant temporal trends.")

    # Visualize a sample watershed with a trend
    trend_ws_ids = trend_summary[trend_summary['has_temporal_trend']]['watershed_id']
    if not trend_ws_ids.empty:
        sample_ws_id = trend_ws_ids.iloc[0]
        print(f"Visualizing sample trend for watershed: {sample_ws_id}")
        
        ws_data = df_with_trends[df_with_trends['watershed_id'] == sample_ws_id]
        ws_lu_data = land_use_df[land_use_df['watershed_id'] == sample_ws_id]
        ws_yearly_avg = ws_data.groupby('year')[['R', 'Tc']].mean()

        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Plot R and Tc yearly averages
        ax1.plot(ws_yearly_avg.index, ws_yearly_avg['R'], 'o-', color='b', label='Yearly Avg R')
        ax1.plot(ws_yearly_avg.index, ws_yearly_avg['Tc'], 's-', color='g', label='Yearly Avg Tc')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Parameter Value (R, Tc)', color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.legend(loc='upper left')
        
        # Plot land use on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(ws_lu_data['year'], ws_lu_data['developed_high_intensity_pct'], '^-', color='r', label='Developed High Intensity %')
        ax2.set_ylabel('Land Use (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
        
        plt.title(f'Temporal Trend Analysis for Watershed {sample_ws_id}')
        fig.tight_layout()
        plt.show()

    return df_with_trends

def identify_storm_outliers_temporal(df, z_score_threshold=2.5):
    """
    Identifies storm outliers, accounting for temporal trends.
    For trend watersheds, Z-score is based on yearly stats.
    For others, it's based on overall stats.
    """
    print(f"\n--- Step 3: Identifying Storm Outliers (Z-score > {z_score_threshold}) ---")
    
    df['R_zscore'] = np.nan
    df['Tc_zscore'] = np.nan

    for ws_id, group in df.groupby('watershed_id'):
        has_trend = group['has_temporal_trend'].iloc[0]
        
        if has_trend:
            # Calculate Z-score relative to yearly stats
            yearly_stats = group.groupby('year').agg(R_mean=('R', 'mean'), R_std=('R', 'std'), Tc_mean=('Tc', 'mean'), Tc_std=('Tc', 'std'))
            group_with_stats = group.join(yearly_stats, on='year')
            
            # Fill std=0 for single-event years to avoid division by zero
            group_with_stats['R_std'] = group_with_stats['R_std'].fillna(0)
            group_with_stats['Tc_std'] = group_with_stats['Tc_std'].fillna(0)

            r_z = (group_with_stats['R'] - group_with_stats['R_mean']) / group_with_stats['R_std']
            tc_z = (group_with_stats['Tc'] - group_with_stats['Tc_mean']) / group_with_stats['Tc_std']
        else:
            # Calculate Z-score relative to overall watershed stats
            r_mean, r_std = group['R'].mean(), group['R'].std()
            tc_mean, tc_std = group['Tc'].mean(), group['Tc'].std()
            r_z = (group['R'] - r_mean) / r_std
            tc_z = (group['Tc'] - tc_mean) / tc_std
        
        df.loc[group.index, 'R_zscore'] = r_z
        df.loc[group.index, 'Tc_zscore'] = tc_z

    # Replace inf/-inf/NaN from division by zero (for single-event years) with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna({'R_zscore': 0, 'Tc_zscore': 0}, inplace=True)

    df['is_outlier'] = (abs(df['R_zscore']) > z_score_threshold) | (abs(df['Tc_zscore']) > z_score_threshold)
    
    outlier_count = df['is_outlier'].sum()
    print(f"Found {outlier_count} outlier storm events after accounting for temporal trends.")
    
    return df[~df['is_outlier']].copy()

def create_final_ml_dataset(df):
    """
    Creates the final aggregated dataset for machine learning.
    - Yearly aggregation for watersheds with trends.
    - Overall aggregation for watersheds without trends.
    """
    print("\n--- Step 4: Creating Final Aggregated Dataset for ML ---")
    final_data = []
    
    for ws_id, group in df.groupby('watershed_id'):
        has_trend = group['has_temporal_trend'].iloc[0]
        all_years = sorted(group['year'].unique())
        
        if has_trend:
            # Aggregate yearly
            yearly_agg = group.groupby('year')[['R', 'Tc']].mean().reset_index()
            yearly_agg.rename(columns={'R': 'R_final', 'Tc': 'Tc_final'}, inplace=True)
            yearly_agg['watershed_id'] = ws_id
            final_data.append(yearly_agg)
        else:
            # Aggregate overall and broadcast to all years
            overall_r = group['R'].mean()
            overall_tc = group['Tc'].mean()
            for year in all_years:
                final_data.append({
                    'watershed_id': ws_id,
                    'year': year,
                    'R_final': overall_r,
                    'Tc_final': overall_tc
                })
                
    final_df = pd.concat(final_data, ignore_index=True) if final_data else pd.DataFrame()
    return final_df


if __name__ == '__main__':
    # --- Configuration & Data Loading ---
    # In a real scenario, load your actual storm and land use data here
    storm_data, land_use_data = simulate_storm_data_with_trends()
    
    print("\nOriginal Data Shape:", storm_data.shape)
    
    # --- Step 1: Filter watersheds with high overall parameter variability ---
    coherent_df, watershed_stats = analyze_watershed_coherence(storm_data, r_cv_threshold=0.6, tc_cv_threshold=0.6)
    
    # --- Step 2: Identify watersheds with significant temporal trends ---
    if not coherent_df.empty:
        df_with_trends = analyze_temporal_and_land_use_trends(coherent_df, land_use_data, p_value_threshold=0.1)
    else:
        df_with_trends = pd.DataFrame()

    # --- Step 3: Filter outlier storm events, respecting temporal trends ---
    if not df_with_trends.empty:
        final_clean_df = identify_storm_outliers_temporal(df_with_trends, z_score_threshold=2.5)
        print(f"\nClean data shape after removing outliers: {final_clean_df.shape}")
    else:
        print("\nNo coherent watersheds found. Halting analysis.")
        final_clean_df = pd.DataFrame()

    # --- Step 4: Generate the final aggregated dataset for the ML model ---
    if not final_clean_df.empty:
        ml_dataset = create_final_ml_dataset(final_clean_df)
        
        print("\n--- Workflow Complete ---")
        print(f"Final ML-ready dataset shape: {ml_dataset.shape}")
        print("\nSample of Final ML Dataset:")
        # Show data from one watershed with a trend and one without
        sample_trend_ws = df_with_trends[df_with_trends['has_temporal_trend']]['watershed_id'].iloc[0]
        sample_notrend_ws = df_with_trends[~df_with_trends['has_temporal_trend']]['watershed_id'].iloc[0]
        print("\n--- Sample: Watershed with Temporal Trend (Yearly Aggregation) ---")
        print(ml_dataset[ml_dataset['watershed_id'] == sample_trend_ws].round(2))
        print("\n--- Sample: Watershed without Temporal Trend (Overall Aggregation) ---")
        print(ml_dataset[ml_dataset['watershed_id'] == sample_notrend_ws].round(2))

