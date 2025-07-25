import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple

def _prepare_data(
    data: Union[pd.DataFrame, pd.Series],
    observed_flow_series: Optional[pd.Series] = None,
    simulated_flow_series: Optional[pd.Series] = None,
    observed_col: str = 'observed_flow',
    simulated_col: str = 'simulated_flow'
) -> Tuple[pd.Series, pd.Series]:
    """
    Internal helper function to parse and validate input data.
    Ensures the output is two pandas Series: observed and simulated flow,
    both with a DatetimeIndex.
    """
    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        if observed_col not in data.columns or simulated_col not in data.columns:
            raise ValueError(
                f"DataFrame must contain '{observed_col}' and '{simulated_col}' columns."
            )
        obs_flow = data[observed_col]
        sim_flow = data[simulated_col]
    elif isinstance(data, pd.Series) and observed_flow_series is not None and simulated_flow_series is not None:
        if not isinstance(data.index, pd.DatetimeIndex):
             # If the first series is not a datetime index, assume it is the datetime series
             datetime_index = pd.to_datetime(data)
             obs_flow = observed_flow_series.copy()
             obs_flow.index = datetime_index
             sim_flow = simulated_flow_series.copy()
             sim_flow.index = datetime_index
        else:
             # Legacy case where data is the observed flow series
             print("Warning: Passing observed flow as the first argument is deprecated. Please use a DataFrame or pass datetime as the first argument.")
             obs_flow = data
             sim_flow = observed_flow_series
             if not isinstance(obs_flow.index, pd.DatetimeIndex) or not isinstance(sim_flow.index, pd.DatetimeIndex):
                 raise ValueError("Input Series must have a DatetimeIndex.")
    else:
        raise TypeError(
            "Invalid input. Provide a DataFrame or three Series (datetime, observed, simulated)."
        )

    return obs_flow.squeeze(), sim_flow.squeeze()


def calculate_single_event_metrics(
    data: Union[pd.DataFrame, pd.Series],
    observed_flow_series: Optional[pd.Series] = None,
    simulated_flow_series: Optional[pd.Series] = None,
    observed_col: str = 'observed_flow',
    simulated_col: str = 'simulated_flow'
) -> Dict[str, float]:
    """
    Processes a single storm event and computes hydrograph metrics.

    Args:
        data (Union[pd.DataFrame, pd.Series]):
            - A pandas DataFrame with a DatetimeIndex and columns for observed and simulated flow.
            - Or, a pandas Series representing the datetime index.
        observed_flow_series (Optional[pd.Series]): The observed flow data. Required if 'data' is a datetime Series.
        simulated_flow_series (Optional[pd.Series]): The simulated flow data. Required if 'data' is a datetime Series.
        observed_col (str): Column name for observed flow if 'data' is a DataFrame.
        simulated_col (str): Column name for simulated flow if 'data' is a DataFrame.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics for the single event.
    """
    obs, sim = _prepare_data(
        data, observed_flow_series, simulated_flow_series, observed_col, simulated_col
    )

    # --- Nash-Sutcliffe Efficiency (NSE) ---
    sum_sq_err = ((obs - sim) ** 2).sum()
    sum_sq_dev_obs = ((obs - obs.mean()) ** 2).sum()
    if sum_sq_dev_obs > 0:
        nse = 1 - (sum_sq_err / sum_sq_dev_obs)
    else:
        nse = np.nan # Cannot be calculated if observed flow is constant

    # --- Peak Flow Metrics ---
    q_obs_peak = obs.max()
    q_sim_peak = sim.max()
    bias_peak_flow = q_sim_peak - q_obs_peak
    if q_obs_peak > 0:
        mape_peak_flow = abs(bias_peak_flow / q_obs_peak) * 100
    else:
        mape_peak_flow = np.nan # Cannot be calculated if observed peak is zero

    # --- Time to Peak Metrics ---
    t_obs_peak = obs.idxmax()
    t_sim_peak = sim.idxmax()
    time_diff_hours = (t_sim_peak - t_obs_peak).total_seconds() / 3600.0
    mae_time_to_peak = abs(time_diff_hours)
    bias_time_to_peak = time_diff_hours

    # --- Volume Metrics ---
    obs_sorted = obs.sort_index()
    if len(obs_sorted.index) > 1:
        time_delta_seconds = (obs_sorted.index[1] - obs_sorted.index[0]).total_seconds()
    else:
        time_delta_seconds = 1.0 # Assume unit time if only one data point
    v_obs = (obs * time_delta_seconds).sum()
    v_sim = (sim * time_delta_seconds).sum()
    bias_volume = v_sim - v_obs
    if v_obs > 0:
        mape_volume = abs(bias_volume / v_obs) * 100
    else:
        mape_volume = np.nan # Cannot be calculated if observed volume is zero

    return {
        'nse': nse,
        'mape_peak_flow': mape_peak_flow,
        'bias_peak_flow': bias_peak_flow,
        'mae_time_to_peak': mae_time_to_peak,
        'bias_time_to_peak': bias_time_to_peak,
        'mape_volume': mape_volume,
        'bias_volume': bias_volume,
    }


def calculate_overall_metrics(
    event_metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
     Calculates the final, aggregated hydrograph performance metrics.

    This function takes a list of metric dictionaries (from Stage 1) for
    multiple events and computes the final averaged metrics.

    Args:
        event_metrics_list (List[Dict]): A list where each dictionary is the
            output of `calculate_single_event_metrics` for an event.

    Returns:
        Dict[str, float]: A dictionary containing the final, aggregated
        performance metrics.
    """
    if not event_metrics_list:
        return {}

    # Convert the list of dictionaries to a DataFrame for easy averaging
    metrics_df = pd.DataFrame(event_metrics_list)

    # Calculate the mean of each column (metric), ignoring NaNs
    averaged_metrics = metrics_df.mean().to_dict()

    # Rename keys for clarity in the final output
    final_metrics = {
        'Average NSE': averaged_metrics.get('nse'),
        'MAPE_Peak Flow (%)': averaged_metrics.get('mape_peak_flow'),
        'Bias_Peak Flow': averaged_metrics.get('bias_peak_flow'),
        'MAE_Time to Peak (hours)': averaged_metrics.get('mae_time_to_peak'),
        'Bias_Time to Peak (hours)': averaged_metrics.get('bias_time_to_peak'),
        'MAPE_Volume (%)': averaged_metrics.get('mape_volume'),
        'Bias_Volume': averaged_metrics.get('bias_volume'),
        'Number of Events': len(event_metrics_list)
    }

    return final_metrics