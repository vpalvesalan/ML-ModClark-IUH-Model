import numpy as np
import pandas as pd

def lyne_hollick_filter(flow_series, alpha=0.925):
    """
    Apply the Lyne-Hollick recursive digital filter with three passes:
    forward, backward, forward, using initial conditions that the first
    (or last, for backward) observation is entirely baseflow.
    
    Parameters:
    -----------
    flow_series : list or pandas.Series
        The streamflow time series data. Must contain no NA values.
    alpha : float
        Recession parameter controlling smoothing (default = 0.925).
        
    Returns:
    --------
    baseflow : np.ndarray
        Estimated baseflow component of the streamflow.
    
    Raises:
    -------
    ValueError:
        If any NA values are present in flow_series.
    """
    # Convert to numpy array
    q = np.asarray(flow_series, dtype=float)
    
    # Check for NA values
    if np.isnan(q).any():
        raise ValueError("Input flow_series contains NA values. Please remove or impute before filtering.")
    
    n = len(q)
    if n == 0:
        return np.array([])
    
    # First pass: forward direction
    b1 = np.zeros(n, dtype=float)
    b1[0] = q[0]  # Assume first observation is all baseflow
    for i in range(1, n):
        b1[i] = alpha * b1[i - 1] + ((1 + alpha) / 2) * (q[i] - q[i - 1])
        if b1[i] > q[i]:
            b1[i] = q[i]  # Ensure baseflow does not exceed total flow
    
    # Second pass: backward direction
    b2 = np.zeros(n, dtype=float)
    b2[-1] = b1[-1]  # Assume last forward-pass baseflow is baseflow at end
    for i in range(n - 2, -1, -1):
        b2[i] = alpha * b2[i + 1] + ((1 + alpha) / 2) * (b1[i] - b1[i + 1])
        if b2[i] > b1[i]:
            b2[i] = b1[i]  # Ensure baseflow does not exceed forward-pass baseflow
    
    # Third pass: forward direction
    baseflow = np.zeros(n, dtype=float)
    baseflow[0] = b2[0]  # Use backward-pass baseflow as initial condition
    for i in range(1, n):
        baseflow[i] = alpha * baseflow[i - 1] + ((1 + alpha) / 2) * (b2[i] - b2[i - 1])
        if baseflow[i] > b2[i]:
            baseflow[i] = b2[i]  # Ensure baseflow does not exceed backward-pass baseflow
    
    return baseflow
