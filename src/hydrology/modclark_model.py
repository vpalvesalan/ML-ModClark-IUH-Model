import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

class ModClarkModel:
    """
    A class to represent and calibrate a ModClark hydrological model for a single storm event.

    This class encapsulates all the necessary data and methods for computing the time-area
    histogram, generating a unit hydrograph, applying losses, simulating runoff, and
    running an optimization to calibrate the model parameters (Tc, R, and losses).

    Attributes:
        df (pd.DataFrame): DataFrame containing the storm event data, including 'quickflow' and 'height'.
        distance_grid (np.ndarray): A 2D grid of flow distances to the watershed outlet.
        cell_area (float): The area of a single grid cell in square meters.
        total_area (float): The total area of the watershed in square meters.
        max_distance (float): The maximum flow distance within the watershed.
        delta_t (float): The time step of the model in seconds.
    """

    def __init__(self, df: pd.DataFrame, distance_grid: np.ndarray, cell_area: float, delta_t: float):
        """Initializes the ModClarkModel with watershed and storm data."""
        self.df = df
        self.distance_grid = distance_grid
        self.cell_area = cell_area
        self.delta_t = delta_t
        
        # Pre-calculate static watershed properties
        self.total_area = np.count_nonzero(distance_grid >= 0) * self.cell_area
        self.max_distance = np.max(distance_grid)

    def _compute_time_area_histogram(self, tc: float) -> np.ndarray:
        """Computes the time-area histogram for a given time of concentration (Tc)."""
        if self.max_distance <= 0:
            return np.array([self.total_area])

        travel_time_grid = tc * (self.distance_grid / self.max_distance)
        num_bins = int(np.ceil(tc / self.delta_t)) + 1
        bin_indices = (travel_time_grid / self.delta_t).astype(int)

        return np.bincount(
            bin_indices.flatten(),
            weights=np.full(bin_indices.size, self.cell_area, dtype=float),
            minlength=num_bins
        )

    def _calculate_modclark_iuh(self, r: float, area_time_histogram: np.ndarray) -> np.ndarray:
        """Computes the scaled ModClark Instantaneous Unit Hydrograph."""
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
            if cumulative_volume + volume_step > 0.995:
                unit_hydrograph = unit_hydrograph[:i+1]
                break
            cumulative_volume += volume_step
            
        scaling_factor = 1.0 / cumulative_volume if cumulative_volume > 0 else 1.0
        return unit_hydrograph * scaling_factor

    def _apply_precipitation_loss(self, initial_loss: float, constant_loss_rate: float) -> np.ndarray:
        """Calculates the excess precipitation hyetograph."""
        excess_precip = self.df['height'].values.copy()
        remaining_initial_loss = initial_loss
        
        for i in range(len(excess_precip)):
            if remaining_initial_loss > 0:
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
        """Performs convolution to generate the direct runoff hydrograph."""
        simulated_hydrograph_depth_rate = np.convolve(excess_precip, unit_hydrograph)[:len(excess_precip)] * self.delta_t
        return simulated_hydrograph_depth_rate * self.total_area

    def _nse_objective_function(self, params: list) -> float:
        """Internal objective function (1 - NSE) for the optimization algorithm."""
        tc, r, initial_loss, constant_loss_rate = params

        # Run the full simulation sequence with the given parameters
        area_histogram = self._compute_time_area_histogram(tc=tc)
        unit_hydrograph = self._calculate_modclark_iuh(r=r, area_time_histogram=area_histogram)
        excess_precip = self._apply_precipitation_loss(initial_loss=initial_loss, constant_loss_rate=constant_loss_rate)
        simulated_flow = self._convolve_uh_with_precipitation(excess_precip=excess_precip, unit_hydrograph=unit_hydrograph)
        
        observed_flow = self.df['quickflow'].values
        if len(simulated_flow) < len(observed_flow):
            simulated_flow = np.pad(simulated_flow, (0, len(observed_flow) - len(simulated_flow)), 'constant')
        else:
            simulated_flow = simulated_flow[:len(observed_flow)]
            
        numerator = np.sum((observed_flow - simulated_flow) ** 2)
        denominator = np.sum((observed_flow - np.mean(observed_flow)) ** 2)
        
        if denominator == 0:
            return 1.0
            
        nse = 1 - (numerator / denominator)
        return 1 - nse

    def run_optimization(self) -> dict:
        """
        Runs the Differential Evolution optimization to calibrate model parameters.

        Returns:
            dict: A dictionary containing the optimization results.
        """
        print("--- 🚀 Starting Optimization ---")
        
        bounds = [
            (3600, 10 * 3600),
            (3600, 20 * 3600),
            (0.0, self.df['height'].sum() * 0.5),
            (0.00001, 0.0001)
        ]

        result = differential_evolution(
            func=self._nse_objective_function,
            bounds=bounds,
            strategy='best1bin',
            maxiter=200,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=True
        )

        print("--- ✅ Optimization Complete ---")
        optimized_params = result.x
        final_nse = 1 - result.fun

        results_dict = {
            "optimized_tc_hr": optimized_params[0] / 3600,
            "optimized_r_hr": optimized_params[1] / 3600,
            "optimized_initial_loss_m": optimized_params[2],
            "optimized_constant_loss_mm_hr": optimized_params[3] * 1000 * 3600,
            "final_nse": final_nse
        }
        
        for key, val in results_dict.items():
            print(f"{key.replace('_', ' ').title():<30}: {val:.4f}")

        return results_dict


def main():
    """
    Main orchestrator function to demonstrate the use of the ModClarkModel class.
    """
    # --- 1. S E T U P ---
    # Load or create the necessary input data for a single storm event.
    
    # 🚨 **ACTION REQUIRED**: Replace this synthetic data with your actual data.
    # For example, load a distance grid from a file:
    # distance_grid = np.load('path/to/your/distance_grid.npy')
    distance_grid = np.random.rand(100, 100) * 5000 # Synthetic grid
    cell_area = 30 * 30  # Area of one cell (e.g., 30m x 30m)
    
    # Create a synthetic DataFrame for demonstration.
    time_steps = 200
    delta_t_minutes = 15
    delta_t_seconds = float(delta_t_minutes * 60)
    
    storm_df = pd.DataFrame({
        'height': np.zeros(time_steps),
        'quickflow': np.zeros(time_steps)
    })
    storm_df.loc[10:19, 'height'] = [0.001, 0.002, 0.005, 0.008, 0.01, 0.007, 0.004, 0.003, 0.002, 0.001]
    storm_df.loc[15:44, 'quickflow'] = [10, 25, 50, 80, 120, 150, 130, 100, 80, 60, 45, 35, 28, 22, 18, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1.5, 1, 0.8, 0.6, 0.4, 0.2]

    # --- 2. M O D E L  E X E C U T I O N ---
    # Instantiate the model for the specific watershed and storm
    model = ModClarkModel(
        df=storm_df,
        distance_grid=distance_grid,
        cell_area=cell_area,
        delta_t=delta_t_seconds
    )

    # Run the optimization
    results = model.run_optimization()


if __name__ == '__main__':
    main()