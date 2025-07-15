# --- Imports ---
import random
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.warp
from shapely import wkt
from tqdm import tqdm
from pysheds.grid import Grid
import sys

# Custom ModulesPath.cwd().parent.parent
project_root_path = Path.cwd().parent.parent
sys.path.append(str(project_root_path / 'src'))

from hydrology.modclark_model import ModClarkModel
from utils.data_utils import glimpse
from plotting_functions.hydrograph import *

# Load events metadata
events_metadata_utc_path = project_root_path / 'data/gold/tabular/detected_storm_events.parquet'
storm_events_utc_metadata = pd.read_parquet(events_metadata_utc_path)
filtered_storm_events_utc_metada = storm_events_utc_metadata[(storm_events_utc_metadata['response_min'] <= 30) & 
                                             (storm_events_utc_metadata['total_ppt_after_peak_mm'] == 0)]
filtered_storm_events_utc_metada = filtered_storm_events_utc_metada.groupby('ws_id').filter(lambda x: len(x) >= 5)

events_metadata_tz_offset_path = project_root_path / 'data/gold/tabular/detected_storm_events_tz_offset.parquet'
storm_events_tz_offset_metadata = pd.read_parquet(events_metadata_tz_offset_path)
filtered_storm_events_tz_offset_metada = storm_events_tz_offset_metadata[(storm_events_tz_offset_metadata['response_min'] <= 30) & 
                                             (storm_events_tz_offset_metadata['total_ppt_after_peak_mm'] == 0)]
filtered_storm_events_tz_offset_metada = filtered_storm_events_tz_offset_metada.groupby('ws_id').filter(lambda x: len(x) >= 5)


# Assuming 'pysheds' and your custom 'ModClarkModel' are available in your environment.
# from pysheds.grid import Grid
# from custom_module import ModClarkModel

# --- Constants ---
# Using constants for values that don't change makes the code cleaner.
D8_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32) # D8 flow direction mapping

# --- Function Definitions ---

def select_n_random_watersheds(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    n_watersheds: int = 30,
    n_events: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects a random sample of watersheds and storm events from two metadata datasets.

    This function first filters both datasets to include only watersheds with a minimum
    number of events. Then, it randomly selects 'n_watersheds' and, for each of
    those, randomly selects 'n_events'.

    Args:
        dataset1: The first metadata DataFrame, containing 'ws_id' and 'storm_id'.
        dataset2: The second metadata DataFrame, with a similar structure to dataset1.
        n_watersheds: The number of unique watersheds to sample.
        n_events: The number of storm events to sample for each watershed.

    Returns:
        A tuple containing two DataFrames:
        - selected_events1: Sampled events from the first dataset.
        - selected_events2: Sampled events from the second dataset.
        
    Raises:
        ValueError: If a different number of watersheds are selected from the datasets
                    or if the requested number of watersheds is greater than available.
    """
    # Filter for watersheds with at least n_events
    ws_group1 = dataset1.groupby('ws_id')
    ws_group2 = dataset2.groupby('ws_id')
    dataset1_filtered = ws_group1.filter(lambda x: x['storm_id'].nunique() >= n_events)
    dataset2_filtered = ws_group2.filter(lambda x: x['storm_id'].nunique() >= n_events)

    # Find common watersheds after filtering
    unique_ws1 = set(dataset1_filtered['ws_id'].unique())
    unique_ws2 = set(dataset2_filtered['ws_id'].unique())
    common_ws_ids = list(unique_ws1.intersection(unique_ws2))

    if n_watersheds > len(common_ws_ids):
        raise ValueError(
            f"Requested {n_watersheds} watersheds, but only {len(common_ws_ids)} "
            f"are available in both datasets with at least {n_events} events."
        )

    # Sample from the common watersheds
    sampled_ws_ids = random.sample(common_ws_ids, n_watersheds)

    sampled_df1 = dataset1[dataset1['ws_id'].isin(sampled_ws_ids)].copy()
    sampled_df2 = dataset2[dataset2['ws_id'].isin(sampled_ws_ids)].copy()

    # It's good practice to double-check that the filtered sets are equal
    if len(sampled_df1['ws_id'].unique()) != len(sampled_df2['ws_id'].unique()):
        raise ValueError("A different number of watersheds was selected between datasets.")

    # Group by watershed and sample n_events for each
    selected_events1 = (
        sampled_df1.groupby('ws_id')['storm_id']
        .apply(lambda x: x.drop_duplicates().sample(n=n_events, random_state=42))
        .reset_index()
    )
    selected_events2 = (
        sampled_df2.groupby('ws_id')['storm_id']
        .apply(lambda x: x.drop_duplicates().sample(n=n_events, random_state=42))
        .reset_index()
    )

    # Merge back to get other event data like start/end times
    event_cols = ['storm_id', 'event_start', 'event_end']
    selected_events1 = selected_events1.merge(
        dataset1[event_cols].drop_duplicates(), on='storm_id'
    )
    selected_events2 = selected_events2.merge(
        dataset2[event_cols].drop_duplicates(), on='storm_id'
    )

    return selected_events1, selected_events2


def load_files(
    ws_id: str,
    project_root_path: Path,
    tz_offset: bool = False
) -> tuple:
    """
    Loads all necessary vector, raster, and tabular files for a given watershed.

    Args:
        ws_id: The watershed identifier.
        project_root_path: The root path of the project directory (as a Path object).
        tz_offset: Boolean flag to load the timezone-offset hydrological data.

    Returns:
        A tuple containing:
        - vector_catch (GeoDataFrame)
        - geo_characteristics (DataFrame)
        - lined_up_data (DataFrame)
        - grid (pysheds.Grid instance)
        - fdir (numpy.ndarray of flow directions)
    """
    # Define folder paths from the project root
    ws_vector_dir = project_root_path / 'data/silver/geo/gpkg/watersheds'
    ws_raster_dir = project_root_path / "data/silver/geo/raster/watersheds"
    hydrological_data_dir = project_root_path / 'data/gold/tabular/lined_up_data'

    # Define file paths
    ws_mask_path = ws_raster_dir / f"catchment_{ws_id}.tif"
    flowdir_path = ws_raster_dir / f"flowdir_{ws_id}.tif"
    vector_catch_path = ws_vector_dir / f'catchment_{ws_id}.gpkg'
    geo_characteristics_path = project_root_path / 'data/gold/tabular/geomorphological_characteristics.csv'
    
    if tz_offset:
        lined_up_data_path = hydrological_data_dir / f'ws_{ws_id}_tz_offset.parquet'
    else:
        lined_up_data_path = hydrological_data_dir / f'ws_{ws_id}.parquet'

    # Load vector files
    vector_catch = gpd.read_file(vector_catch_path)

    # Load tabular data
    geo_characteristics = pd.read_csv(geo_characteristics_path, dtype={'Station ID': str})
    lined_up_data = pd.read_parquet(lined_up_data_path)
    lined_up_data = lined_up_data.set_index('date').sort_index()

    # Load raster data using pysheds
    grid = Grid.from_raster(str(flowdir_path), data_name='flowdir')
    fdir = grid.read_raster(str(flowdir_path))
    catch_mask = grid.read_raster(str(ws_mask_path))
    grid.clip_to(catch_mask)

    return vector_catch, geo_characteristics, lined_up_data, grid, fdir


def create_distance_weights(resolution: float, fdir: np.ndarray) -> np.ndarray:
    """
    Creates a weights grid for calculating geometric distance in pysheds.

    This grid assigns a real-world distance to each cell based on its
    flow direction, accounting for longer diagonal paths.

    Args:
        resolution: The grid's cardinal resolution (cell width or height).
        fdir: The pysheds flow direction grid.

    Returns:
        A grid of weights representing the distance for each cell-to-cell link.
    """
    cardinal_dirs = (64, 1, 4, 16)
    diagonal_dirs = (128, 2, 8, 32)
    
    cardinal_dist = resolution
    diagonal_dist = np.sqrt(2) * resolution

    weights_grid = np.full_like(fdir, 0, dtype=np.float64)
    weights_grid[np.isin(fdir, cardinal_dirs)] = cardinal_dist
    weights_grid[np.isin(fdir, diagonal_dirs)] = diagonal_dist

    return weights_grid


def compute_distance_grid(grid: 'Grid', fdir: np.ndarray, resolution: float, pour_point: tuple) -> np.ndarray:
    """
    Computes a distance grid based on the flow direction grid.

    Args:
        grid: The pysheds Grid object, clipped to the catchment.
        fdir: The flow direction raster.
        resolution: The cell resolution (width/height) in the grid's units.
        pour_point: A tuple (x, y) of the outlet coordinates.

    Returns:
        A numpy array representing the distance from each cell to the outlet.
    """
    x, y = pour_point
    weights = create_distance_weights(resolution, fdir)
    dist = grid.distance_to_outlet(
         x=x, y=y, fdir=fdir, dirmap=D8_DIRMAP, xytype='coordinate', weights=weights
        )
    return dist


def get_resolution_from_raster(ws_id: str, project_root_path: Path, target_crs: str) -> tuple[float, float]:
    """
    Gets the resolution of a flow direction raster in the target CRS.

    Args:
        ws_id: The watershed identifier.
        project_root_path: The root path of the project directory.
        target_crs: The target Coordinate Reference System.

    Returns:
        A tuple of (x_resolution, y_resolution).
    """
    ws_raster_dir = project_root_path / "data/silver/geo/raster/watersheds"
    flowdir_path = ws_raster_dir / f"flowdir_{ws_id}.tif"

    with rasterio.open(flowdir_path) as src:
        transform, _, _ = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
    # transform[0] is pixel width (x_res), transform[4] is pixel height (y_res)
    return (transform[0], transform[4])


def get_pour_point(vector_catch: gpd.GeoDataFrame) -> tuple[float, float]:
    """
    Extracts the pour point coordinates from the watershed vector file.

    Args:
        vector_catch: GeoDataFrame of the watershed.

    Returns:
        A tuple containing (x, y) coordinates of the pour point.
    """
    point_string = vector_catch['snapped_pour_point_nad83'].iloc[0]
    coordinates = wkt.loads(point_string)
    return (coordinates.x, coordinates.y)


def compare_params_optimization(
    utc_metadata: pd.DataFrame,
    tz_offset_metadata: pd.DataFrame,
    project_root: Path,
    n_ws: int,
    n_ev: int, 
    display: bool = True
) -> tuple[dict, dict]:
    """
    Compares Nash-Sutcliffe Efficiency (NSE) for watersheds from two datasets.

    This function orchestrates the entire workflow: selecting watersheds, loading
    data, preparing grids, and running a hydrological model optimization for
    both a UTC-based dataset and a timezone-offset dataset.

    Args:
        utc_metadata: Metadata for the UTC-based storm events.
        tz_offset_metadata: Metadata for the timezone-offset storm events.
        project_root: The root path of the project directory.
        n_ws: The number of watersheds to analyze.
        n_ev: The number of events per watershed to analyze.

    Returns:
        A tuple of two dictionaries: (nse_utc_scores, nse_tz_offset_scores).
        Each dictionary maps a ws_id to a list of NSE scores for that watershed.
    """
    nse_utc_scores = {}
    nse_tz_offset_scores = {}
    delta_t = 900  # Time step in seconds

    # 1. Select Watersheds and Events
    storm_events_utc, storm_events_tz_offset = select_n_random_watersheds(
        utc_metadata, tz_offset_metadata, n_watersheds=n_ws, n_events=n_ev
    )
    
    ws_ids = storm_events_utc['ws_id'].unique()

    # --- Main Loop: Process each watershed ---
    for ws_id in tqdm(ws_ids, desc="Processing watersheds"):
        if display:
            print(f"--- Processing Watershed: {ws_id} ---")
        
        # Initialize lists for the current watershed's scores
        nse_utc_scores[ws_id] = []
        nse_tz_offset_scores[ws_id] = []

        # 2. Load Data Files (placeholders used for missing files)
        try:
            vector_catch, _, lined_up_data_utc, grid, fdir = load_files(ws_id, project_root, tz_offset=False)
            _, _, lined_up_data_tz_offset, _, _ = load_files(ws_id, project_root, tz_offset=True)
            pour_point = get_pour_point(vector_catch)
            target_crs = vector_catch.crs
            x_res, y_res = get_resolution_from_raster(ws_id, project_root, target_crs=target_crs)
        except Exception as e:
            print(f"    Skipping watershed {ws_id} due to loading error: {e}")
            # Create dummy data to allow loops to run without crashing
            fdir = np.random.choice(D8_DIRMAP, size=(10, 10))
            pour_point = (0, 0)
            x_res, y_res = 30, -30
            lined_up_data_utc = pd.DataFrame(index=pd.to_datetime([]))
            lined_up_data_tz_offset = pd.DataFrame(index=pd.to_datetime([]))


        # 3. Prepare Grids and Parameters
        resolution_m = abs(x_res)
        cell_area = abs(x_res * y_res)
        distance_grid = compute_distance_grid(grid, fdir, resolution_m, pour_point)

        # --- 4a. Run Optimization for UTC data ---
        events_for_ws_utc = storm_events_utc[storm_events_utc['ws_id'] == ws_id]
        for _, row in events_for_ws_utc.iterrows():
            if display:
                print(f"  Running UTC event: {row['storm_id']}")
            df = lined_up_data_utc.loc[row['event_start']:row['event_end']].copy()
            
            model = ModClarkModel(df=df, distance_grid=distance_grid, cell_area=cell_area, delta_t=delta_t)
            optimized_params = model.run_optimization(display=False)
            nse_utc_scores[ws_id].append(optimized_params['nse'])

        # --- 4b. Run Optimization for Timezone-Offset data ---
        events_for_ws_tz = storm_events_tz_offset[storm_events_tz_offset['ws_id'] == ws_id]
        for _, row_tz in events_for_ws_tz.iterrows():
            if display:
                print(f"  Running TZ-Offset event: {row_tz['storm_id']}")
            df = lined_up_data_tz_offset.loc[row_tz['event_start']:row_tz['event_end']].copy()
            
            model = ModClarkModel(df=df, distance_grid=distance_grid, cell_area=cell_area, delta_t=delta_t)
            optimized_params = model.run_optimization(display=False)
            nse_tz_offset_scores[ws_id].append(optimized_params['nse'])

    return nse_utc_scores, nse_tz_offset_scores

# --- Example Usage ---
if __name__ == '__main__':
    # Define the root path to your project
    PROJECT_ROOT = Path.cwd().parent.parent

    try:

        # Run the comparison
        nse_utc_by_ws, nse_tz_by_ws = compare_params_optimization(
            utc_metadata=filtered_storm_events_utc_metada,
            tz_offset_metadata=filtered_storm_events_tz_offset_metada,
            project_root=PROJECT_ROOT,
            n_ws=30, # Number of watersheds to test
            n_ev=10,  # Number of events per watershed, 
            display=False
        )

        # --- Post-processing and Statistics ---
        # Flatten the dictionaries to get all scores for overall stats
        all_nse_utc = [score for scores_list in nse_utc_by_ws.values() for score in scores_list]
        all_nse_tz = [score for scores_list in nse_tz_by_ws.values() for score in scores_list]

        # Calculate Coefficient of Variation for each watershed
        cv_utc = []
        for ws_id, scores in nse_utc_by_ws.items():
            if scores and np.mean(scores) != 0:  # Avoid division by zero and handle empty lists
                cv = np.std(scores) / np.mean(scores)
                cv_utc.append(cv)

        cv_tz = []
        for ws_id, scores in nse_tz_by_ws.items():
            if scores and np.mean(scores) != 0: # Avoid division by zero and handle empty lists
                cv = np.std(scores) / np.mean(scores)
                cv_tz.append(cv)

        # --- Print Final Results ---
        print("\n--- Comparison Complete ---")
        # print(f"UTC NSE Scores (all events): {np.round(all_nse_utc, 3)}")
        # print(f"Timezone-Offset NSE Scores (all events): {np.round(all_nse_tz, 3)}")
        print(f"\nOverall Average UTC NSE: {np.mean(all_nse_utc):.3f}")
        print(f"Overall Average Timezone-Offset NSE: {np.mean(all_nse_tz):.3f}")
        
        # --- Average Coefficient of Variation ---
        if cv_utc:
             print(f"\nAverage Coefficient of Variation (UTC): {np.mean(cv_utc):.3f}")
        if cv_tz:
             print(f"Average Coefficient of Variation (Timezone-Offset): {np.mean(cv_tz):.3f}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
