# -------------------------------
# Import Libraries & Define Helpers
# -------------------------------
import numpy as np
# Assume a DE implementation is available or user-defined
# from differential_evolution import DifferentialEvolution
# For file loading and plotting, import necessary libraries accordingly

# -------------------------------
# Step 1: Data Preparation
# -------------------------------
def load_data():
    """
    Load the following:
    - Precipitation events: dictionary with keys as event IDs and values as cell-wise time series.
    - Observed flow data: dictionary with keys as event IDs and values as the observed flow time series.
    - Grid data: a dictionary or 2D array with cell IDs and their relative distance to the outlet.
    """
    precipitation_events = load_precipitation_data()  # Implement this loader
    observed_flows = load_observed_flow_data()          # Implement this loader
    grid_distance = load_grid_data()                    # Implement this loader to get cell distances
    return precipitation_events, observed_flows, grid_distance

# -------------------------------
# Step 2: Calculate Precipitation Excess (Grid-Cell Wise)
# -------------------------------
def calculate_precip_excess(event_precip, initial_loss, constant_loss, grid_distance):
    """
    For each grid cell in event_precip:
      - Subtract initial loss first; once exhausted, subtract constant loss each time step.
    Returns a dictionary where each key is cell ID and value is the effective precipitation time series.
    """
    effective_precip = {}  # Dictionary: {cell_id: effective_time_series}
    for cell in grid_distance:  # For each cell
        effective_precip[cell] = []
        loss_remaining = initial_loss  # Initial loss for the cell
        for P in event_precip[cell]:
            if loss_remaining > 0:
                if P <= loss_remaining:
                    effective_value = 0
                    loss_remaining -= P
                else:
                    effective_value = P - loss_remaining
                    loss_remaining = 0
            else:
                effective_value = max(P - constant_loss, 0)
            effective_precip[cell].append(effective_value)
    return effective_precip

# -------------------------------
# Helper: Compute Travel Time for a Cell Given Tc
# -------------------------------
def compute_travel_time(cell_distance, Tc):
    """
    Compute travel time for a cell.
    Example formulation: travel_time = cell_distance / Tc
    (Ensure Tc and distance are in consistent units.)
    """
    return cell_distance / Tc

def get_time_index(travel_time, simulation_time_vector, Delta_t):
    """
    Map a travel time to the closest time step index in simulation_time_vector.
    Delta_t is the time step length (e.g., in minutes or hours).
    """
    # Simple version: index = int(travel_time / Delta_t)
    return int(travel_time / Delta_t)

# -------------------------------
# Step 3: Simulation Module (ModClark Model)
# -------------------------------
def modclark_simulation(parameters, event_precip, grid_distance, simulation_time_vector):
    """
    Simulate flow for one event using the ModClark transformation.
    
    Parameters:
        parameters: [Tc, R, initial_loss, constant_loss]
        event_precip: Dictionary for one event {cell_id: time series of precipitation}
        grid_distance: Dictionary {cell_id: distance to outlet}
        simulation_time_vector: Array of simulation time steps
        
    Returns:
        simulated_flow: Simulated flow time series (array) corresponding to simulation_time_vector.
    """
    # Unpack parameters
    Tc, R, initial_loss, constant_loss = parameters
    # Define Delta_t: ensure unit consistency, e.g., if simulation_time_vector is in minutes:
    Delta_t = compute_delta_t(simulation_time_vector)  # e.g., 15 (minutes)
    
    # 3a. Compute effective precipitation (grid-cell wise)
    effective_precip = calculate_precip_excess(event_precip, initial_loss, constant_loss, grid_distance)
    
    # 3b. Construct the translation hydrograph I_i (aggregated over cells)
    I = np.zeros(len(simulation_time_vector))
    for cell in grid_distance:
        cell_distance = grid_distance[cell]
        travel_time = compute_travel_time(cell_distance, Tc)  # travel time for this cell
        delay_index = get_time_index(travel_time, simulation_time_vector, Delta_t)
        cell_precip_ts = effective_precip[cell]  # effective precipitation time series for cell
        
        # Allocate each cell's precipitation values to the translation hydrograph using the delay
        for t, value in enumerate(cell_precip_ts):
            alloc_index = t + delay_index
            if alloc_index < len(simulation_time_vector):
                I[alloc_index] += value  # Sum contributions from all cells

    # 3c. Calculate the IUH ordinates using the recursive formulation
    # Compute routing coefficient c using equation: c = Delta_t / (R + 0.5*Delta_t)
    c = Delta_t / (R + 0.5 * Delta_t)
    
    # Initialize arrays for IUH and U (unit hydrograph ordinates)
    IUH = np.zeros_like(I)
    U = np.zeros_like(I)
    
    # Initialization: Assume IUH[0] = c*I[0] (or assign an impulse value if simulating a unit impulse)
    IUH[0] = c * I[0]
    U[0] = IUH[0]  # Optionally, set the first ordinate to IUH[0]
    
    # For i from 1 to end: Use the recursive equation:
    #   IUH[i] = c * I[i] + (1-c) * IUH[i-1]
    # Then compute:
    #   U[i] = (IUH[i-1] + IUH[i]) / 2
    for i in range(1, len(I)):
        IUH[i] = c * I[i] + (1 - c) * IUH[i - 1]
        U[i] = 0.5 * (IUH[i - 1] + IUH[i])
    
    # 3d. Convolve the translation hydrograph I with U to get simulated flow Q.
    #   Q[i] = sum_{j=0}^{i} I[j] * U[i-j]
    # Using convolution and matching output length:
    simulated_flow = np.convolve(I, U)[:len(simulation_time_vector)]
    
    # 3e. Truncate or adjust the tail if necessary (for instance, if the flow becomes negligible).
    simulated_flow = truncate_simulated_flow(simulated_flow)
    
    return simulated_flow

# -------------------------------
# Helper: Compute Delta_t from Time Vector
# -------------------------------
def compute_delta_t(simulation_time_vector):
    """
    Compute the time step Delta_t from simulation_time_vector.
    Assume simulation_time_vector is in uniform steps (e.g., minutes)
    """
    # For example, if simulation_time_vector = [0, 15, 30, ...]
    if len(simulation_time_vector) >= 2:
        return simulation_time_vector[1] - simulation_time_vector[0]
    else:
        return 15  # Default to 15 minutes if undefined

# -------------------------------
# Helper: Truncate Simulated Flow (Optional)
# -------------------------------
def truncate_simulated_flow(flow, threshold=1e-3):
    """
    Truncate the hydrograph when flow values fall below a threshold.
    For example, remove tail portions where flow < threshold (or after a fixed number of steps).
    """
    # One simple method: find the last index where flow >= threshold.
    indices = np.where(flow >= threshold)[0]
    if len(indices) > 0:
        last_index = indices[-1] + 1  # +1 to include this point
        return flow[:last_index]
    else:
        return flow

# -------------------------------
# Step 4: Objective Function for a Single Event
# -------------------------------
def compute_NSE(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE):
        NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)
    """
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (numerator / denominator)

def objective_function(parameters, event_id, precipitation_events, observed_flows, grid_distance, simulation_time_vector):
    """
    For a single event, run the simulation and return the negative NSE (to be minimized by DE).
    parameters: candidate vector [Tc, R, initial_loss, constant_loss]
    event_id: ID for the event to be calibrated
    """
    # Retrieve event-specific precipitation and observed flow data
    event_precip = precipitation_events[event_id]
    obs_flow = observed_flows[event_id]
    
    # Simulate the flow using the ModClark simulation module
    sim_flow = modclark_simulation(parameters, event_precip, grid_distance, simulation_time_vector)
    
    # Compute NSE
    nse_value = compute_NSE(obs_flow, sim_flow)
    
    # Since DE minimizes, return negative NSE
    return -nse_value

# -------------------------------
# Step 5: Differential Evolution Optimization for a Single Event
# -------------------------------
def run_DE_for_event(event_id, precipitation_events, observed_flows, grid_distance, simulation_time_vector):
    """
    Run the Differential Evolution (DE) algorithm for one event.
    Candidate parameters: [Tc, R, initial_loss, constant_loss]
    
    Note: Loss parameters have predefined bounds. For Tc and R, you can define default bounds or derive them.
    """
    # Define parameter bounds (example; adjust as needed)
    # For initial_loss and constant_loss, bounds are predefined.
    bounds = {
        'Tc': (0.1, 10.0),  # Example lower/upper bounds for travel time scaling parameter (units consistent with grid distances)
        'R': (0.01, 20.0),  # Example bounds for storage coefficient
        'initial_loss': (predefined_init_loss_lower, predefined_init_loss_upper),
        'constant_loss': (predefined_const_loss_lower, predefined_const_loss_upper)
    }
    # Convert bounds to an ordered list [Tc, R, initial_loss, constant_loss]
    bounds_list = [bounds['Tc'], bounds['R'], bounds['initial_loss'], bounds['constant_loss']]
    
    # Define the DE objective function wrapper for the given event
    def de_objective(candidate_vector):
        return objective_function(candidate_vector, event_id, precipitation_events, observed_flows, grid_distance, simulation_time_vector)
    
    # Set DE hyperparameters (example values)
    de_options = {
        'pop_size': 50,
        'mutation': 0.8,
        'crossover': 0.9,
        'max_iter': 200
    }
    
    # Run the DE algorithm
    best_parameters, best_obj_value = DifferentialEvolution(de_objective, bounds_list, de_options).optimize()
    
    # The best objective value is negative NSE; convert back:
    best_NSE = -best_obj_value
    return best_parameters, best_NSE

# -------------------------------
# Step 6: Main Function - Loop Over Events for Individual Optimization
# -------------------------------
def main():
    # Load data
    precipitation_events, observed_flows, grid_distance = load_data()
    
    # Define simulation time vector (e.g., an array of times at 15-min intervals)
    simulation_time_vector = create_simulation_time_vector()  # Implement as needed
    
    # For each event, run optimization using DE
    optimized_results = {}
    for event_id in precipitation_events.keys():
        best_params, best_NSE = run_DE_for_event(event_id, precipitation_events, observed_flows, grid_distance, simulation_time_vector)
        optimized_results[event_id] = {'parameters': best_params, 'NSE': best_NSE}
        print("Event:", event_id)
        print("Optimized Parameters: Tc= {:.3f}, R= {:.3f}, Initial Loss= {:.3f}, Constant Loss= {:.3f}".format(
              best_params[0], best_params[1], best_params[2], best_params[3]))
        print("Achieved NSE =", best_NSE, "\n")
    
    # Save or return the optimized_results for further analysis
    return optimized_results

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
