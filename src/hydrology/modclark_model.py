from ast import Tuple
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from typing import Tuple

from shapely import area

class ModClarkModel:
    """
    A class to represent and calibrate a ModClark hydrological model for a single storm event.

    This class encapsulates all the necessary data and methods for computing the time-area
    histogram, generating a unit hydrograph, applying losses, simulating runoff, and
    running an optimization to calibrate the model parameters (Tc, R, and losses).

    Attributes:
        df (pd.DataFrame): DataFrame containing the storm event data, including 'quickflow' and 'height'.
        distance_grid (np.ndarray): A 2D NumPy array where each cell value is the
                                        flow distance from that cell to the watershed outlet [m].
                                        This grid should not contain negative values (e.g., NaN or
                                        no-data values should be handled beforehand).
        cell_area (float): The area of a single grid cell in square meters.
        total_area (float): The total area of the watershed in square meters.
        max_distance (float): The maximum flow distance within the watershed in meters.
        delta_t (float): The time step of the model in seconds.
    """

    def __init__(self, df: pd.DataFrame, distance_grid: np.ndarray, cell_area: float, delta_t: float):
            """
            Initializes the ModClarkModel with watershed and storm data.

            This constructor also pre-processes the distance grid to filter out
            non-finite values for computational efficiency.
            """
            self.df = df
            self.distance_grid = distance_grid
            self.cell_area = cell_area
            self.delta_t = delta_t


            valid_mask = np.isfinite(self.distance_grid)
            self.valid_distances = self.distance_grid[valid_mask]
            self.max_distance = np.max(self.valid_distances) if self.valid_distances.size > 0 else 0
            self.total_area = self.valid_distances.size * self.cell_area

    def _compute_time_area_histogram(self, tc: float) -> np.ndarray:
        """
        Computes the time-area histogram for a watershed based on travel times.

        This method translates a spatial grid of flow distances to the outlet into a
        temporal distribution of contributing areas. It calculates the travel time for each
        grid cell using the basin's time of concentration (Tc) and then discretizes these
        times into bins, summing the area in each bin.

        Args:
            tc (float): The basin's time of concentration in seconds. This parameter is
                        being optimized.

        Returns:
            np.ndarray: A 1D NumPy array representing the time-area histogram. The index
                        corresponds to the time step, and the value is the total contributing
                        area [m²] for that interval.
        """

        if self.max_distance <= 0:
            return np.array([self.total_area])

        travel_time_grid = tc * (self.valid_distances / self.max_distance)
        num_bins = int(np.ceil(tc / self.delta_t)) + 1
        bin_indices = (travel_time_grid / self.delta_t).astype(int)

        return np.bincount(
            bin_indices.flatten(),
            weights=np.full(self.valid_distances.size, self.cell_area, dtype=float),
            minlength=num_bins
        )

    def _calculate_modclark_iuh(self, r: float, area_time_histogram: np.ndarray) -> np.ndarray:
        """
        Computes the ModClark Instantaneous Unit Hydrograph (IUH) and scales it.

        This method implements the ModClark model's recursive routing approach. It translates
        the inflow, derived from the watershed's time-area histogram, through a conceptual
        linear reservoir at the basin outlet to generate the IUH. The final unit hydrograph
        is truncated when its cumulative volume exceeds 0.995 and then scaled to ensure
        the total volume is exactly 1.0.

        Args:
            r (float): The storage coefficient of the linear reservoir in seconds. This
                    parameter governs the attenuation of the hydrograph.
            area_time_histogram (np.ndarray): A 1D NumPy array representing the distribution of
                                            watershed area contributing to flow over time. The
                                            index corresponds to the time step, and the value
                                            is the contributing area in square meters.
        Returns:
            np.ndarray: A 1D NumPy array of the final, scaled unit hydrograph ordinates [T⁻¹].
        """
        c = self.delta_t / (r + 0.5 * self.delta_t)
        inflow = area_time_histogram / (self.total_area * self.delta_t) if self.total_area > 0 else np.zeros_like(area_time_histogram)
        
        iuh = np.zeros_like(inflow)
        for i in range(1, len(inflow)):
            iuh[i] = c * inflow[i] + (1 - c) * iuh[i-1]
            
        unit_hydrograph = np.zeros_like(inflow)
        for i in range(1, len(iuh)):
            unit_hydrograph[i] = (iuh[i-1] + iuh[i]) / 2
            
        cumulative_volume = 0.0
        for i in range(1, len(unit_hydrograph)):
            volume_step = (unit_hydrograph[i] + unit_hydrograph[i-1]) / 2 * self.delta_t
            if cumulative_volume + volume_step > 0.999:
                unit_hydrograph = unit_hydrograph[:i+1]
                break
            cumulative_volume += volume_step
            
        scaling_factor = 1.0 / cumulative_volume if cumulative_volume > 0 else 1.0
        return unit_hydrograph * scaling_factor

    def _apply_precipitation_loss(self, initial_loss: float, constant_loss_rate: float) -> np.ndarray:
        """
        Calculates the excess precipitation hyetograph using an initial and constant loss model.

        This method simulates surface runoff generation by first satisfying an initial abstraction
        (e.g., interception, depression storage) from the start of the storm. Once this initial
        volume is met, a continuous, constant rate of loss (e.g., infiltration) is subtracted
        from the remaining rainfall intensity for the rest of the event.

        Args:
            initial_loss (float): The total amount of initial abstraction [m] that must be
                                satisfied before any runoff can occur.
            constant_loss_rate (float): The continuous rate of precipitation loss [m/s] that
                                        occurs after the initial loss is met.
        Returns:
            np.ndarray: A 1D NumPy array of the excess precipitation [m] for each time step.
        """
        excess_precip = self.df['height'].values.copy()
        remaining_initial_loss = initial_loss

        rain_mask = self.df['height'].values > 0
        
        for i in np.where(rain_mask)[0]:
            if remaining_initial_loss <= 0:
                break

            precip_at_step = excess_precip[i]
            if precip_at_step >= remaining_initial_loss:
                excess_precip[i] -= remaining_initial_loss
                remaining_initial_loss = 0
            else:
                remaining_initial_loss -= precip_at_step
                excess_precip[i] = 0
        
                    
        constant_loss_per_step = constant_loss_rate * self.delta_t
        return np.maximum(0, excess_precip - constant_loss_per_step)

    def _convolve_uh_with_precipitation(self, excess_precip: np.ndarray, unit_hydrograph: np.ndarray) -> np.ndarray:
        """
        Performs convolution to generate the direct runoff hydrograph.

        This function simulates the watershed's runoff response by convolving the excess
        precipitation hyetograph (the input signal) with the unit hydrograph (the watershed's
        response function). The resulting depth-rate hydrograph is then scaled by the
        watershed area to produce the final volumetric discharge hydrograph.

        Args:
            excess_precip (np.ndarray): A 1D NumPy array of the excess precipitation [m] for
                                        each time step.
            unit_hydrograph (np.ndarray): A 1D NumPy array of the unit hydrograph ordinates [T⁻¹].

        Returns:
            np.ndarray: A 1D NumPy array of the simulated volumetric discharge [m³/s].
        """
        simulated_hydrograph_depth_rate = np.convolve(excess_precip, unit_hydrograph)
        return simulated_hydrograph_depth_rate * self.total_area

    def _nse_objective_function(self, params: list) -> float:
        """
        Objective function for the optimization, calculating 1 - NSE.

        This method serves as the bridge between the optimization algorithm and the
        hydrological model. It takes a set of model parameters, runs a full simulation
        to generate a hydrograph, and then evaluates its goodness-of-fit against observed
        data using the Nash-Sutcliffe Efficiency (NSE) coefficient. The function returns
        1 - NSE because optimization algorithms typically perform minimization.

        Args:
            params (list): A list containing the model parameters to be optimized, in the
                        order: [Tc (s), R (s), initial_loss (m), constant_loss_rate (m/s)].

        Returns:
            float: The value of 1 - NSE. A perfect model fit would return 0.
        """
        tc, r, initial_loss, constant_loss_rate = params

        # Run the full simulation sequence with the given parameters
        area_histogram = self._compute_time_area_histogram(tc=tc)
        unit_hydrograph = self._calculate_modclark_iuh(r=r, area_time_histogram=area_histogram)
        excess_precip = self._apply_precipitation_loss(initial_loss=initial_loss, constant_loss_rate=constant_loss_rate)
        simulated_flow = self._convolve_uh_with_precipitation(excess_precip=excess_precip, unit_hydrograph=unit_hydrograph)
        
        observed_flow = self.df['quickflow'].values
        if len(simulated_flow) > len(observed_flow):
            simulated_flow = simulated_flow[:len(observed_flow)]
        elif len(simulated_flow) < len(observed_flow):
            simulated_flow = np.pad(simulated_flow, (0, len(observed_flow) - len(simulated_flow)), 'constant')
            
        numerator = np.sum((observed_flow - simulated_flow) ** 2)
        denominator = np.sum((observed_flow - np.mean(observed_flow)) ** 2)
        
        if denominator == 0:
            return 1.0
            
        nse = 1 - (numerator / denominator)
        return 1 - nse

    def run_optimization(self, tc_bounds = (900, 10*3600), r_bounds= (900, 20*3600), geo_char: Tuple[float, float] = None, dynamic_ppt_loss_bounds: float = False, workers=1, display: bool = True) -> dict:
        """Runs a Differential Evolution optimization to calibrate hydrologic model parameters.

        This method calibrates four key parameters: Time of Concentration ($T_c$), 
        Storage Coefficient ($R$), initial precipitation loss, and constant 
        precipitation loss rate.

        The search bounds for $T_c$ and $R$ can be dynamically set based on watershed
        geomorphological characteristics using the Kirpich formula. Otherwise, fixed 
        default bounds are used. The bounds for precipitation loss can also be 
        dynamically estimated from the input storm data.

        Args:
            tc_bounds (Tuple[float, float], optional): The bounds for the Time of Concentration ($T_c$) in seconds.
                Defaults to `(900, 10*3600)` which corresponds to 15 minutes to 10 hours.
            r_bounds (Tuple[float, float], optional): The bounds for the Storage Coefficient ($R$) in seconds.
                Defaults to `(900, 20*3600)` which corresponds to 15 minutes to 20 hours.
            geo_char (Tuple[float, float], optional): A tuple containing the 
                watershed's `(basin_length_m, slope_10_85)`. `basin_length_m` is the 
                main channel length in meters, and `slope_10_85` is the average 
                channel slope as a decimal (e.g., 0.02). If provided, these are 
                used to dynamically calculate $T_c$ and $R$ bounds. Defaults to `None`.
            dynamic_ppt_loss_bounds (bool, optional): If `True`, dynamically calculates 
                the bounds for initial precipitation loss based on the timing of 
                rainfall and quickflow onset. If `False`, uses a wider default 
                range. Defaults to `False`.
            display (bool, optional): If `True`, prints optimization progress and 
                final results to the console. Defaults to `True`.
            workers (int, optional): The number of parallel workers to use for the
                optimization. Defaults to `1`, meaning no parallelization.

        Returns:
            dict: A dictionary containing the calibrated parameters and model performance:
                - `tc_hr`: Optimized Time of Concentration ($T_c$) in hours.
                - `r_hr`: Optimized Storage Coefficient ($R$) in hours.
                - `initial_loss_mm`: Optimized initial loss in millimeters.
                - `constant_loss_mm_hr`: Optimized constant loss rate in mm/hour.
                - `nse`: The Nash-Sutcliffe Efficiency (NSE) of the calibrated model.
        """
        if display:
            print("--- 🚀 Starting Optimization ---")

        if geo_char is not None:
            basin_length_m, slope_10_85 = geo_char

        # --- Set parameter search bounds ---
        if geo_char is not None:
            # Dynamically set bounds using Kirpich formula and rules of thumb
            if display:
                print("Using dynamic bounds based on watershed characteristics.")

            # Kirpich formula (Tc in hours)
            tc_kirpich_hr = 0.000323 * (basin_length_m ** 0.77) * (slope_10_85 ** -0.385)

            # Tc bounds in seconds
            tc_min_sec = 0.4 * tc_kirpich_hr * 3600
            tc_max_sec = 2.5 * tc_kirpich_hr * 3600
            tc_bounds = (tc_min_sec, tc_max_sec)

            # R bounds in seconds, derived from Tc bounds
            r_min_sec = 1.0 * tc_min_sec
            r_max_sec = 5.0 * tc_max_sec
            r_bounds = (r_min_sec, r_max_sec)
        
        else:
            # Use fixed, default bounds if no characteristics are provided
            if display:
                print("Using fixed, default search bounds.")
        bounds = [
            tc_bounds,
            r_bounds
        ]

        # Initial ppt loss bounds in meters
        total_ppt = self.df['height'].sum()
        if dynamic_ppt_loss_bounds:
            ppt_start_iloc = (self.df['height'] > 0).idxmax()

            if (self.df['quickflow'] > 0).any():
                quickflow_start_iloc = (self.df['quickflow'] > 0).idxmax()
            else:
                quickflow_start_iloc = self.df.index[-1]
            min_initial_loss = self.df.loc[ppt_start_iloc:quickflow_start_iloc, 'height'].sum()
            max_initial_loss = min_initial_loss * 2 if min_initial_loss > 0 else self.df['height'].sum() * 0.1
            max_initial_loss = min(max_initial_loss, total_ppt * 0.75)

            if max_initial_loss < min_initial_loss:
                min_initial_loss = total_ppt * 0.01
                max_initial_loss = total_ppt * 0.99
            initial_loss_bounds = (min_initial_loss, max_initial_loss)
        else:
            max_initial_loss = 0
            initial_loss_bounds = (0, total_ppt)

        # Constant ppt loss rate bounds in meters per second
        ppt_time_steps = np.sum(self.df['height']>0)
        if ppt_time_steps < 0:
            raise ValueError("Total precipitation in the storm event should be greater than 0.")
        max_constant_loss_rate = (total_ppt - max_initial_loss)  / (ppt_time_steps * self.delta_t)
        constant_loss_bounds = (0, max_constant_loss_rate) 

        bounds += [initial_loss_bounds, constant_loss_bounds]

        # --- Run the optimization ---
        result = differential_evolution(
            func=self._nse_objective_function,
            bounds=bounds,
            strategy='best1bin',
            maxiter=500,
            popsize=50,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            workers=workers,
            disp=display # Control console output
        )

        # --- Process and return results ---
        if display:
            print("--- ✅ Optimization Complete ---")

        optimized_params = result.x
        final_nse = result.fun

        results_dict = {
            "tc_hr": optimized_params[0] / 3600,
            "r_hr": optimized_params[1] / 3600,
            "initial_loss_mm": optimized_params[2]*1000,
            "constant_loss_mm_hr": optimized_params[3] * 1000 * 3600,
            "nse": 1-final_nse
        }
        
        if display:
            for key, val in results_dict.items():
                print(f"{key.replace('_', ' ').title():<30}: {val:.4f}")

        return results_dict

    def simulate(self, tc: float, r: float, initial_loss: float, constant_loss_rate: float) -> np.ndarray:
            """
            Runs a single simulation with a given set of parameters.

            This public method is intended for use after optimization to generate the
            final simulated hydrograph with the optimal parameters.

            Args:
                tc (float): The time of concentration in seconds.
                r (float): The storage coefficient in seconds.
                initial_loss (float): The initial loss in meters.
                constant_loss_rate (float): The constant loss rate in m/s.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                    - simulated_flow (np.ndarray): The simulated hydrograph [m³/s].
                    - excess_precip (np.ndarray): The excess precipitation hyetograph [m].
                    - unit_hydrograph (np.ndarray): The unit hydrograph ordinates [T⁻¹].
                    - area_histogram (np.ndarray): The time-area histogram [m²].
            """
            # Run the full simulation sequence with the provided parameters
            area_histogram = self._compute_time_area_histogram(tc=tc)
            unit_hydrograph = self._calculate_modclark_iuh(r=r, area_time_histogram=area_histogram)
            excess_precip = self._apply_precipitation_loss(initial_loss=initial_loss, constant_loss_rate=constant_loss_rate)
            simulated_flow = self._convolve_uh_with_precipitation(excess_precip=excess_precip, unit_hydrograph=unit_hydrograph)

            # Align the final hydrograph to the length of the observed data
            observed_flow = self.df['quickflow'].values
            if len(simulated_flow) > len(observed_flow):
                simulated_flow = simulated_flow[:len(observed_flow)]
            elif len(simulated_flow) < len(observed_flow):
                simulated_flow = np.pad(simulated_flow, (0, len(observed_flow) - len(simulated_flow)), 'constant')

            return simulated_flow, excess_precip, unit_hydrograph, area_histogram
