from pathlib import Path
import geopandas as gpd
import pandas as pd
from pysheds.grid import Grid
import os
from time import sleep
import numpy as np
import rasterio
import rasterio.warp
from tqdm import tqdm
from shapely import wkt
import random

# Project root path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path.cwd().parent.parent


# --- Constants ---
D8_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

# --- Helper Functions (Moved from main script) ---

def create_distance_weights(resolution: float, fdir: np.ndarray) -> np.ndarray:
    """Creates a weights grid for calculating geometric distance in pysheds."""
    cardinal_dirs = (64, 1, 4, 16)
    diagonal_dirs = (128, 2, 8, 32)
    cardinal_dist = resolution
    diagonal_dist = np.sqrt(2) * resolution
    weights_grid = np.full_like(fdir, 0, dtype=np.float64)
    weights_grid[np.isin(fdir, cardinal_dirs)] = cardinal_dist
    weights_grid[np.isin(fdir, diagonal_dirs)] = diagonal_dist
    return weights_grid

def compute_distance_grid(grid: 'Grid', fdir: np.ndarray, resolution: float, pour_point: tuple) -> np.ndarray:
    """Computes a distance grid based on the flow direction grid."""
    x, y = pour_point
    weights = create_distance_weights(resolution, fdir)
    dist = grid.distance_to_outlet(
        x=x, y=y, fdir=fdir, dirmap=D8_DIRMAP, xytype='coordinate', weights=weights
    )
    return dist

def get_pour_point(vector_catch: gpd.GeoDataFrame) -> tuple[float, float]:
    """Extracts the pour point coordinates."""
    point_string = vector_catch['snapped_pour_point_nad83'].iloc[0]
    coordinates = wkt.loads(point_string)
    return (coordinates.x, coordinates.y)

def get_resolution_from_raster(ws_id: str, project_root_path: Path, target_crs: str) -> tuple[float, float]:
    """Gets the resolution of a flow direction raster."""
    flowdir_path = project_root_path / f"data/silver/geo/raster/watersheds/flowdir_{ws_id}.tif"
    with rasterio.open(flowdir_path) as src:
        transform, _, _ = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
    return (transform[0], transform[4])

# --- Main Preprocessing Function ---

def preprocess_and_save_distance_grid(ws_id: str, project_root: Path, display=False):
    """
    Computes and saves the distance-to-outlet grid for a watershed as a .npy file.
    """
    if display:
        print(f"Preprocessing distance grid for watershed: {ws_id}...")
    try:
        # Define I/O paths
        ws_raster_dir = project_root / "data/silver/geo/raster/watersheds"
        ws_vector_dir = project_root / 'data/silver/geo/gpkg/watersheds'
        output_dir = project_root / "data/gold/geo/raster/watersheds"
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        flowdir_path = ws_raster_dir / f"flowdir_{ws_id}.tif"
        ws_mask_path = ws_raster_dir / f"catchment_{ws_id}.tif"
        vector_catch_path = ws_vector_dir / f'catchment_{ws_id}.gpkg'
        output_path = output_dir / f"distance_grid_{ws_id}.npy"

        # 1. Load necessary geo-data
        vector_catch = gpd.read_file(vector_catch_path)
        grid = Grid.from_raster(str(flowdir_path), data_name='flowdir')
        fdir = grid.read_raster(str(flowdir_path), dtype=np.float32)
        catch_mask = grid.read_raster(str(ws_mask_path), dtype=np.float32)
        grid.clip_to(catch_mask)

        # 2. Get parameters for distance calculation
        pour_point = get_pour_point(vector_catch)
        target_crs = vector_catch.crs
        x_res, _ = get_resolution_from_raster(ws_id, project_root, target_crs)
        resolution_m = abs(x_res)

        # 3. Compute the distance grid
        distance_grid = compute_distance_grid(grid, fdir, resolution_m, pour_point)

        # 4. Save the final grid as a .npy file
        np.save(output_path, distance_grid)
        if display:
            print(f"  -> Successfully saved to {output_path}")

    except Exception as e:
        print(f"  -> ERROR: Failed to preprocess {ws_id}: {e}")

if __name__ == '__main__':
    # --- Setup ---
    # Define the project root and output directory
    output_dir = PROJECT_ROOT / "data/gold/geo/raster/watersheds"
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Get a list of all watershed IDs that need processing ---
    utc_metadata_path = PROJECT_ROOT / 'data/gold/tabular/detected_storm_events.parquet'
    utc_metadata = pd.read_parquet(utc_metadata_path)
    utc_metadata['ws_id'] = utc_metadata['storm_id'].str.split('_').str[0]
    all_watershed_ids = utc_metadata['ws_id'].unique()

    # --- 2. Find watersheds that have already been processed ---
    # List files in the output directory and extract the ws_id from each filename
    processed_ws_ids = {
        f.replace('distance_grid_', '').replace('.npy', '') 
        for f in os.listdir(output_dir) if f.startswith('distance_grid_') and f.endswith('.npy')
    }

    # --- 3. Determine which watersheds to process now ---
    # Exclude the already processed watersheds from the main list
    ws_to_process = [ws_id for ws_id in all_watershed_ids if ws_id not in processed_ws_ids]
    random.shuffle(ws_to_process)

    print(f"Found {len(all_watershed_ids)} total watersheds.")
    print(f"Found {len(processed_ws_ids)} already processed grids. Skipping them.")
    print(f"--- Starting preprocessing for {len(ws_to_process)} remaining watersheds ---")

    # --- 4. Run preprocessing only on the remaining watersheds ---
    if ws_to_process:
        for ws_id in tqdm(ws_to_process, desc="Preprocessing Watersheds"):
            preprocess_and_save_distance_grid(ws_id, PROJECT_ROOT)
            sleep(0.5)
    else:
        print("All watershed already preprocessed.")
    print("--- Preprocessing Complete ---")