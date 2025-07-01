import numpy as np
import pandas as pd

def lyne_hollick_filter(flow_series, alpha=0.925):
    """
    Apply the Lyne-Hollick recursive digital filter with three passes,
    forward, backward, forward, to separate quickflow.

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
    """
    
    # Convert to numpy array
    q = np.asarray(flow_series, dtype=float)

    # Check for NA values
    if np.isnan(q).any():
        raise ValueError("Input flow_series contains NA values. Please remove or impute before filtering.")

    n = len(q)
    if n == 0:
        return np.array([])

    # The filter calculates quickflow directly.
    # Let's rename the variables to reflect this: qf for quickflow.

    # First pass (forward) to estimate quickflow
    qf1 = np.zeros(n, dtype=float)
    qf1[0] = 0  # Assume no quickflow at the start
    for i in range(1, n):
        # The Lyne-Hollick filter equation for quickflow
        qf1[i] = alpha * qf1[i - 1] + ((1 + alpha) / 2) * (q[i] - q[i - 1])
        if qf1[i] < 0:
            qf1[i] = 0
        if qf1[i] > q[i]:  # Quickflow cannot exceed total flow
            qf1[i] = q[i]

    # Second pass (backward) to refine quickflow
    qf2 = np.zeros(n, dtype=float)
    qf2[n - 1] = 0 # Assume no quickflow at the end
    for i in range(n - 2, -1, -1):
        qf2[i] = alpha * qf2[i + 1] + ((1 + alpha) / 2) * (qf1[i] - qf1[i + 1])
        if qf2[i] < 0:
            qf2[i] = 0
        if qf2[i] > qf1[i]: # Refined quickflow cannot exceed previous estimate
            qf2[i] = qf1[i]

    # Third pass (forward) for final quickflow estimate
    quickflow = np.zeros(n, dtype=float)
    quickflow[0] = 0 # Assume no quickflow at the start
    for i in range(1, n):
        quickflow[i] = alpha * quickflow[i - 1] + ((1 + alpha) / 2) * (qf2[i] - qf2[i - 1])
        if quickflow[i] < 0:
            quickflow[i] = 0
        if quickflow[i] > qf2[i]: # Final quickflow cannot exceed previous estimate
            quickflow[i] = qf2[i]
            
    return quickflow
