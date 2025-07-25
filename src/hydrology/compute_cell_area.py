# This file computes the cell area for each watershed and saves it locally
import rasterio
import rasterio.warp
from pathlib import Path
import geopandas as gpd
import pandas as pd


def compute_cell_area_from_raster(ws_id: str, PROJECT_ROOT: Path, target_crs: str) -> tuple[float, float]:
    """Gets the resolution of a flow direction raster."""
    flowdir_path = PROJECT_ROOT / f"data/silver/geo/raster/watersheds/flowdir_{ws_id}.tif"
    with rasterio.open(flowdir_path) as src:
        transform, _, _ = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
    return abs(transform[0]) * abs(transform[4])

def get_target_crs(PROJECT_ROOT: Path, ws_id: str) -> str:
    """
    Returns the target CRS for a given watershed.
    """
    catch_vector_path = PROJECT_ROOT / f"data/gold/geo/gpkg/watersheds/catchment_{ws_id}.gpkg"
    catch_vector_crs = gpd.read_file(catch_vector_path).crs
    return catch_vector_crs  

def get_wsid_list(PROJECT_ROOT: Path) -> list[str]:
    """
    Returns a list of watershed IDs from the project root directory.
    """
    wsid_list = []
    for wsid in PROJECT_ROOT.glob("data/silver/geo/raster/watersheds/flowdir_*.tif"):
        wsid_list.append(wsid.stem.split('_')[1])
    return wsid_list

def main():
    
    source_file_path = Path(__file__)
    PROJECT_ROOT = source_file_path.parent.parent.parent
    ws_id_list = get_wsid_list(PROJECT_ROOT)
    areas = []
  
    print("\n--- Computing cell areas for each watershed... ---")
    for ws_id in ws_id_list:
        try: 
            target_crs = get_target_crs(PROJECT_ROOT, ws_id)
            cell_area = compute_cell_area_from_raster(ws_id, PROJECT_ROOT, target_crs)
            areas.append(cell_area)
        except Exception as e:
            areas.append(None)
            continue

    print("\n--- Cell areas computed successfully! ---")
    areas_df = pd.DataFrame({'ws_id': ws_id_list, 'cell_area_m_squared': areas})
    areas_df_path = PROJECT_ROOT / "data/gold/tabular/cell_area.csv"
    areas_df.to_csv(areas_df_path, index=False)
    print(f"\n--- Cell areas saved to {areas_df_path} ---")

if __name__ == "__main__":
    main()