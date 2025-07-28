from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    
    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    
    Returns:
    float: MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def bias(y_true, y_pred):
    """
    Calculate the bias between true and predicted values.
    
    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    
    Returns:
    float: Bias value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_pred - y_true)

def calculate_direct_metrics(y_true, y_pred):
    """
    Calculate various regression metrics.
    
    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    
    Returns:
    dict: Dictionary containing R2, RMSE, MAPE, and Bias.
    """
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'Bias': bias(y_true, y_pred)
    }
    return metrics

def convert_fractional_hours_to_hm(fractional_hours):
    """
    Converts a decimal number representing hours into hours and minutes.

    Args:
        fractional_hours (float): The time in fractional hours (e.g., 2.75).

    Returns:
        tuple: A tuple containing two integers (hours, minutes).
    """
    hours = int(fractional_hours)
    minutes_decimal = (fractional_hours - hours) * 60
    minutes = int(minutes_decimal)
    return hours, abs(minutes)

import os
import sys
from pathlib import Path
file_path = Path(os.path.abspath(__file__))
sys.path.append(str(os.path.dirname(file_path.parent.parent)))
from hydrology.modclark_model import ModClarkModel

def calculate_hydrological_metrics(tc_model, r_model, data: Dict[pd.DataFrame, pd.Datafra, np.array, float, float, float]) -> Tuple[Dict, pd.DataFrame]:
    """
    Calculate hydrological metrics from a dictionary of DataFrames and parameters.

    Args:
        tc_model (object): The model used for Tc predictions.
        r_model (object): The model used for R predictions.
        data (Dict[pd.DataFrame, float, float]): Dictionary where keys are watershed IDs
        and values: 
            - pd.DataFrame: geomorphological characteristic for the watershed
            - pd.DataFrame: historical tabular data (storm event) for the specific storm event
            - np.array: distance grid for the watershed
            - cell area: cell area in square meters of distance grid
            - initial_loss: initial precipitation loss in m
            - constant_loss: constant loss in m/s.

    Returns:
        Tuple[Dict, pd.DataFrame]: A tuple containing a dictionary of average metrics
            for all events in dictionary and a DataFrame with single metric for each event.
    """

    for ws_id, (geo_char, event, distance_grid, cell_area, initial_loss, constant_loss) in data.items():
        
        
        hours, minutes = convert_fractional_hours_to_hm(mean)
        metrics[key] = {
            'Mean (h)': hours,
            'Mean (min)': minutes,
            'Std (h)': std
        }
        results_df = results_df.append(metrics[key], ignore_index=True)

    return metrics, results_df