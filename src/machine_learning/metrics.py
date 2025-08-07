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