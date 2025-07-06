import pandas as pd
import numpy as np
from datetime import timedelta

def detect_storm_events(df, time_step_minutes=15):
    """
    Identifies storm events from timeseries data based on precipitation and quickflow criteria.

    This function processes a DataFrame containing hydrological data to detect discrete
    storm events. An event is defined by a period of rainfall followed by a runoff
    response (quickflow), subject to several filtering conditions.

    The event detection logic is as follows:
    1.  Identifies all contiguous periods of non-zero quickflow.
    2.  For each period, it applies the following mandatory filters:
        a. The quickflow must be sustained for a minimum duration (at least 12 consecutive
           time steps, e.g., 3 hours for 15-minute data).
        b. The quickflow must be initiated by recent rainfall, starting no more than
           3 hours after the last recorded precipitation.
        c. The peak quickflow during the event must be significant relative to the
           total streamflow, specifically at least 50% of the peak discharge
           observed during the same event period.
    3.  The boundaries of a valid event are defined from the start of the causative
       rainfall (determined by a preceding dry period) until the cessation of the 
       resulting quickflow.
    4.  For each valid event, metadata is computed, including total precipitation,
       duration, peak flows, and antecedent/post-event moisture conditions.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a DatetimeIndex or a 'date' column that can be converted to one.
        It must contain the following columns:
        - 'discharge': Total streamflow measurement.
        - 'quickflow': The portion of streamflow considered rapid runoff.
        - 'height': Total precipitation depth for each time step.
        - 'ppt_stn_*': Optional columns for individual precipitation stations, used for
          calculating spatial variability.
    time_step_minutes : int, optional
        The time resolution of the data in minutes (default is 15). This is used
        to calculate event durations in hours.

    Returns
    -------
    list[dict]
        A list of dictionaries, where each dictionary represents a detected storm event
        and contains the following metadata:
        - 'start': Event start timestamp.
        - 'end': Event end timestamp.
        - 'month': The month in which the event started.
        - 'total_precipitation': Sum of precipitation during the event.
        - 'event_duration_min': Total duration of the event in hours.
        - 'peak_quickflow': Maximum quickflow value during the event.
        - 'peak_discharge': Maximum discharge value during the event.
        - 'temporal_variability': Standard deviation of precipitation during rainfall periods.
        - 'spatial_variability': Mean standard deviation of precipitation across stations.
        - 'ppt_14d', 'ppt_7d', ...: Antecedent precipitation totals.
        - 'ppt_post_24h', 'ppt_post_6h', ...: Post-event precipitation totals.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # --- Constants for event criteria ---
    MIN_QUICKFLOW_STEPS = 12
    MAX_TIME_SINCE_PPT = timedelta(hours=3)
    PEAK_FLOW_RATIO = 0.30
    DRY_PERIOD_STEPS = 3  # Number of consecutive zero-precipitation steps to define event start

    # Identify columns for spatial variability calculation
    ppt_cols = [col for col in df.columns if col.startswith('ppt_stn_')]

    # --- Step 1: Identify all contiguous blocks of non-zero quickflow ---
    is_qf = df['quickflow'] > 0
    qf_starts = df.index[is_qf & ~is_qf.shift(1).fillna(False)]
    qf_ends = df.index[is_qf & ~is_qf.shift(-1).fillna(False)]

    if len(qf_ends) < len(qf_starts):
        qf_ends = qf_ends.append(pd.Index([df.index[-1]]))

    events = []

    # --- Step 2: Filter quickflow blocks and define valid events ---
    for qf_start_time, qf_end_time in zip(qf_starts, qf_ends):
        qf_block = df.loc[qf_start_time:qf_end_time]

        # --- Filter 1: Quickflow must be sustained for at least 12 time steps ---
        if len(qf_block) < MIN_QUICKFLOW_STEPS:
            continue

        # --- Filter 3: Quickflow must start within 3 hours of recent precipitation ---
        search_window_start = qf_start_time - MAX_TIME_SINCE_PPT
        precip_search_df = df.loc[search_window_start:qf_start_time]
        if precip_search_df['height'].sum() == 0:
            continue

        # --- Define Event Boundaries ---
        event_end_time = qf_end_time
        last_ppt_time = precip_search_df[precip_search_df['height'] > 0].index.max()
        last_ppt_idx = df.index.get_loc(last_ppt_time)

        zero_ppt_count = 0
        event_start_idx = last_ppt_idx
        for i in range(last_ppt_idx, -1, -1):
            if df.iloc[i]['height'] == 0:
                zero_ppt_count += 1
            else:
                zero_ppt_count = 0  # Reset counter if rain
            
            if zero_ppt_count >= DRY_PERIOD_STEPS:
                event_start_idx = i + 1  # Event starts after the dry spell
                break
        else:
            event_start_idx = 0  # Reached the beginning of the dataframe

        event_start_time = df.index[event_start_idx]
        
        # Prevent overlapping events
        if events and event_start_time <= events[-1]['event_end']:
            continue
            
        event_df = df.loc[event_start_time:event_end_time]

        # --- Filter 2: Peak quickflow must be at least 50% of peak discharge ---
        peak_qf = event_df['quickflow'].max()
        peak_discharge = event_df['discharge'].max()

        if peak_discharge == 0 or (peak_qf / peak_discharge) < PEAK_FLOW_RATIO:
            continue
        
        # --- Filter 4: Event should be in Jan or Feb ---
        if event_start_time.month in [1, 2]:
            continue
        
        # --- Event is valid, compute metadata ---
        # Isolate the portion of the event where precipitation is recorded
        ppt_in_event = event_df[event_df['height'] > 0]
        if not ppt_in_event.empty:
            first_ppt_time = ppt_in_event.index.min()
            last_ppt_time = ppt_in_event.index.max()
            ppt_period_df = event_df.loc[first_ppt_time:last_ppt_time]

            temporal_var = ppt_period_df['height'].std()
        else:
            temporal_var = 0

        total_ppt_mm = event_df['height'].sum() *1000
        ppt_dutation_min = len(ppt_period_df) * time_step_minutes if not ppt_period_df.empty else 0
        event_duration_min = len(event_df) * time_step_minutes
        spatial_var = event_df[ppt_cols].std(axis=1).mean() if ppt_cols else np.nan

        # Get total precipitation after peak quickflow
        peak_qf_time = event_df['quickflow'].idxmax()
        total_ppt_after_peak_mm = event_df.loc[peak_qf_time:, 'height'].sum() * 1000

        # Response after the first precipitation
        response_min = (first_ppt_time - event_start_time).total_seconds() / 60
        
        antecedent = {}
        antecedent_end_time = event_start_time - timedelta(minutes=time_step_minutes)
        for period in ['14d', '7d', '3d', '24h', '6h', '3h']:
            win_start = antecedent_end_time - pd.to_timedelta(period)
            antecedent[f'ppt_prior_{period}'] = df.loc[win_start:antecedent_end_time, 'height'].sum()

        post = {}
        post_start_time = event_end_time + timedelta(minutes=time_step_minutes)
        for period in ['3h', '6h', '24h']:
            win_end = post_start_time + pd.to_timedelta(period)
            post[f'ppt_post_{period}'] = df.loc[post_start_time:win_end, 'height'].sum()
            
        events.append({
            'event_start': event_start_time,
            'event_end': event_end_time,
            'month': event_start_time.month,
            'ppt_start': first_ppt_time,
            'ppt_end': last_ppt_time,
            'total_precipitation_mm': total_ppt_mm,
            'total_ppt_after_peak_mm': total_ppt_after_peak_mm,
            'response_min': response_min,
            'ppt_dutation_min': ppt_dutation_min,
            'event_duration_min': event_duration_min,
            'peak_quickflow': peak_qf,
            'ppt_temporal_variability': temporal_var,
            'ppt_spatial_variability': spatial_var,
            'n_ppt_stations': len(ppt_cols),
            **antecedent,
            **post
        })

    return events