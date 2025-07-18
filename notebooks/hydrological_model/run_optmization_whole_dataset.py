import os
import pandas as pd
from pathlib import Path
import rasterio
import rasterio.warp
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import sys

script_path = os.path.abspath(__file__)
PROJECT_ROOT = Path(script_path).parent.parent.parent

# Custom Modules
sys.path.append(str(PROJECT_ROOT / 'src'))
from hydrology.modclark_model import ModClarkModel

def load_distance_grid(ws_id: str, project_root: Path) -> np.ndarray:
    """
    Loads a pre-computed distance grid from a .npy file.
    
    Args:
        ws_id: The watershed identifier.
        project_root: The root path of the project directory.

    Returns:
        A numpy array of the distance grid.
    
    Raises:
        FileNotFoundError: If the pre-computed file does not exist.
    """
    distance_grid_path = project_root / f"data/gold/geo/raster/watersheds/distance_grid_{ws_id}.npy"
    if not distance_grid_path.exists():
        raise FileNotFoundError(
            f"Distance grid not found: {distance_grid_path}. "
            "Please run the `preprocess_grids.py` script first."
        )
    return np.load(distance_grid_path)


def get_resolution_from_raster(ws_id: str, PROJECT_ROOT: Path, target_crs: str) -> tuple[float, float]:
    """Gets the resolution of a flow direction raster."""
    flowdir_path = PROJECT_ROOT / f"data/silver/geo/raster/watersheds/flowdir_{ws_id}.tif"
    with rasterio.open(flowdir_path) as src:
        transform, _, _ = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
    return (transform[0], transform[4])


def load_files(
    ws_id: str,
    PROJECT_ROOT: Path
    ) -> tuple:
    """
    Loads all necessary vector, raster, and tabular files for a given watershed.

    Args:
        ws_id: The watershed identifier.
        PROJECT_ROOT: The root path of the project directory (as a Path object).
        tz_offset: Boolean flag to load the timezone-offset hydrological data.

    Returns:
        A tuple containing:
        - vector_catch (GeoDataFrame)
        - lined_up_data_utc (DataFrame)
        - lined_up_data_tz_offset (DataFrame)
    """
    # Define folder paths from the project root
    ws_vector_dir = PROJECT_ROOT / 'data/silver/geo/gpkg/watersheds'
    hydrological_data_dir = PROJECT_ROOT / 'data/gold/tabular/lined_up_data'

    # Define file paths
    vector_catch_path = ws_vector_dir / f'catchment_{ws_id}.gpkg'
    geo_characteristics_path = PROJECT_ROOT / 'data/gold/tabular/geomorphological_characteristics.csv'
    
    lined_up_data_path_tz_offset = hydrological_data_dir / f'ws_{ws_id}_tz_offset.parquet'
    lined_up_data_path_utc = hydrological_data_dir / f'ws_{ws_id}.parquet'

    # Load vector files
    vector_catch = gpd.read_file(vector_catch_path)

    # Load tabular data
    lined_up_data_tz_offset = pd.read_parquet(lined_up_data_path_tz_offset)
    lined_up_data_tz_offset = lined_up_data_tz_offset.set_index('date').sort_index()
    geo_characteristcs = pd.read_csv(geo_characteristics_path, dtype={'Station ID':str})
    geo_characteristcs = geo_characteristcs.set_index('Station ID')

    lined_up_data_utc = pd.read_parquet(lined_up_data_path_utc)
    lined_up_data_utc = lined_up_data_utc.set_index('date').sort_index()

    return vector_catch, lined_up_data_utc, lined_up_data_tz_offset, geo_characteristcs

# Main comparison function ---
def batch_optimization(
    storms_metada: pd.DataFrame,
    project_root: Path,
    utc: bool,
    display: bool = True
) -> tuple[dict, dict, dict]: # Added dict for dummy, adjust if needed
    """
    Optmize the whole set of events for the chosen dataset
    """
    scores = {}
    delta_t = 900  # Time step in seconds

    ws_ids = storms_metada['ws_id'].unique()
    for ws_id in ws_ids:
        if display:
            print(f"--- Processing Storm Events for Watershed: {ws_id} ---")

        scores[ws_id] = [] 

        try:
            # Load hydrological data and vector file to get metadata
            if utc:
                if display:
                    print("UTC dataset selected")
                vector_catch, lined_up_data, _ , geo_char = load_files(ws_id, project_root)
            else:
                if display:
                    print("Time zone offset dataset selected")
                vector_catch, _, lined_up_data, geo_char = load_files(ws_id, project_root)
            
            # Get necessary parameters
            target_crs = vector_catch.crs
            x_res, y_res = get_resolution_from_raster(ws_id, project_root, target_crs=target_crs)
            
            # --- Load the pre-computed distance grid ---
            distance_grid = load_distance_grid(ws_id, project_root)
            cell_area = abs(x_res * y_res)

            
        except Exception as e:
            print(f"    Skipping watershed {ws_id} due to loading error: {e}")
            # Create dummy data to allow loops to run without crashing
            distance_grid = np.zeros((10, 10))
            cell_area = 900
            lined_up_data = pd.DataFrame(index=pd.to_datetime([]))

        events = storms_metada[storms_metada['ws_id'] == ws_id]

        # --- Run Optimizations ---
        for _, row in events.iterrows():
            df = lined_up_data.loc[row['event_start']:row['event_end']].copy()
            
            model = ModClarkModel(df=df, distance_grid=distance_grid, cell_area=cell_area, delta_t=delta_t)
            if display:
                print(f"    Running optmization for storm event: {row['storm_id']}")
                print(f"            Running computed boundaries for ppt loss")
            optimized_params = model.run_optimization(tc_bounds = (900, 20*3600), r_bounds= (0.1*3600, 20*3600), dynamic_ppt_loss_bounds = False, display=display) 
    
            # Append a dictionary with both storm_id and NSE
            optimized_params['storm_id'] = row['storm_id']
            scores[ws_id].append(optimized_params)

    return scores


def main():
    scores_path = PROJECT_ROOT / "data/gold/tabular/optmized_params/optmized_whole_tz_dataset.csv"
    distance_grid_path = PROJECT_ROOT / "data/gold/geo/raster/watersheds"
    df_path = PROJECT_ROOT / "data/gold/tabular/detected_storm_events_tz_offset_filtered.parquet"

    df = pd.read_parquet(df_path)
    if distance_grid_path.is_dir():
        files = distance_grid_path.glob('*.npy')
        distance_grid_ws_ids = {f.stem.replace('distance_grid_', '') for f in files}
        
        filtered_storm_events_tz_offset_metada = df[df['ws_id'].isin(distance_grid_ws_ids)]
    else:
        filtered_storm_events_tz_offset_metada = df

    if scores_path.exists():
        existing_scores = pd.read_csv(scores_path)
        existing_storm_ids = set(existing_scores['storm_id'].unique())
        filtered_storm_events_tz_offset_metada = filtered_storm_events_tz_offset_metada[~filtered_storm_events_tz_offset_metada['storm_id'].isin(existing_storm_ids)]
        print(f"Skipping {len(existing_storm_ids)} events already optmized.\n")

    batch_size = 10
    num_rows = len(filtered_storm_events_tz_offset_metada)
    loop = list(range(0, num_rows, batch_size))

    print(f"Start optmization of {num_rows} events.")

    for i in tqdm(loop, f"Processing Sortm Events by batch size of {batch_size}..."):
        df = filtered_storm_events_tz_offset_metada.iloc[i:i + batch_size]
        try:
            scores = batch_optimization(
                storms_metada=df,
                project_root=PROJECT_ROOT,
                utc=False,
                display=False
            )

            scores_data = []
            for ws_id, events in scores.items():
                for event_data in events:
                    scores_data.append({
                        'ws_id': ws_id,
                        'storm_id': event_data['storm_id'],
                        'tc_hr': event_data['tc_hr'],
                        'r_hr': event_data['r_hr'],
                        'initial_loss_mm': event_data['initial_loss_mm'],
                        'constant_loss_mm_hr': event_data['constant_loss_mm_hr'],
                        'nse': event_data['nse']
                    })
            df_scores = pd.DataFrame(scores_data)

            # --- Save result for each batch ---
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            if scores_path.exists():
                df_scores.to_csv(scores_path, mode='a', index=False, header=False)
            else:
                df_scores.to_csv(scores_path, index=False)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
    print("Optimization completed.")