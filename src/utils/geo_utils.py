import re
import rasterio
import rasterio.warp
import numpy as np
from pysheds.view import Raster, ViewFinder
from pyproj import CRS, Transformer


def extract_coordinates_from_point_string(string):
    """
    Extracts longitude and latitude from a string formatted as 'POINT (longitude latitude)'.
    
    Parameters:
    string (str): The input string containing the coordinates.
    
    Returns:
    tuple: A tuple containing the longitude and latitude as floats.
    """
    # Regular expression to match the POINT format
    pattern = r'POINT\s*\((.*?)\s*\)'
    
    # Match the pattern against the string
    match = re.match(pattern, string)
    
    if match:
        # Get the content inside parentheses
        coords = match.group(1).split()
        longitude = float(coords[0])
        latitude = float(coords[1])
        return longitude, latitude
    else:
        raise ValueError("No match found in the input string.")
    
def reproject_raster_in_memory(input_raster_path: str, target_crs: str):
    """
    Reprojects a raster to a new CRS entirely in memory.

    Args:
        input_raster_path (str): The file path to the input raster.
        target_crs (str): The target Coordinate Reference System (e.g., 'EPSG:32618').

    Returns:
        Raster: pysheds Raster object containing the reprojected data.
    """
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    with rasterio.open(input_raster_path) as src:
        # Calculate the transform and dimensions of the reprojected raster
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Create an empty numpy array to hold the reprojected data
        reprojected_array = np.empty((height, width), dtype=src.profile['dtype'])

        # Perform the reprojection
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=reprojected_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=rasterio.warp.Resampling.bilinear
        )

        view_finder = ViewFinder(
            affine=transform,
            shape=reprojected_array.shape,
            crs=target_crs,
            nodata=src.nodata
        )

    return Raster(reprojected_array, viewfinder=view_finder, metadata={'dirmap':dirmap, 'routing':'d8'})

def reproject_point(point_wkt: str, source_crs: str, target_crs: str) -> tuple:
    """Projects a WKT point to a new CRS."""
    coords_str = point_wkt.upper().replace('POINT', '').strip().strip('()')
    x_str, y_str = coords_str.split()
    x, y = float(x_str), float(y_str)

    transformer = Transformer.from_crs(
        crs_from=CRS(source_crs),
        crs_to=CRS(target_crs),
        always_xy=True
    )

    projected_x, projected_y = transformer.transform(x, y)

    return (projected_x, projected_y)