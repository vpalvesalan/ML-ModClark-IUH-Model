import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import shape
from scipy.ndimage import distance_transform_edt
import math
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from typing import List



# ---------- FUNCTION FOR PREPROCESSING ---------- #
def clean_watershed_gdf(gdf, keep_all_parts=False):
    """
    Cleans a watershed GeoDataFrame by removing interior rings (holes) and optionally simplifying MultiPolygons.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the watershed geometry. Must have exactly one row.

    keep_all_parts : bool, default False
        - If False: keeps only the largest polygon part from a MultiPolygon.
        - If True: keeps all polygon parts, but removes interior rings from each.

    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with the cleaned geometry.
    """
    
    if len(gdf) != 1:
        raise ValueError("GeoDataFrame must contain exactly one row.")

    # Access the geometry
    geom = gdf.geometry.iloc[0]

    # Process based on geometry type
    if isinstance(geom, MultiPolygon):
        if keep_all_parts:
            # Remove holes from all parts
            cleaned_parts = [Polygon(part.exterior) for part in geom.geoms]
            clean_geom = MultiPolygon(cleaned_parts)
        else:
            # Keep only largest part, remove its holes
            largest = max(geom.geoms, key=lambda p: p.area)
            clean_geom = Polygon(largest.exterior)
    elif isinstance(geom, Polygon):
        # Remove holes
        clean_geom = Polygon(geom.exterior)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    # Create a copy of the GeoDataFrame
    cleaned_gdf = gdf.copy()
    cleaned_gdf.at[cleaned_gdf.index[0], 'geometry'] = clean_geom

    return cleaned_gdf

import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np

def preprocess_mrlc(mrlc_path, watershed_gdf, dem_resolution, output_path=None):
    """
    Preprocess MRLC raster by clipping to watershed, reprojecting to watershed CRS, resampling, 
    and handling NoData values.
    
    Parameters:
    -----------
    mrlc_path : str
        Path to MRLC raster.
    watershed_gdf : geopandas.GeoDataFrame
        Watershed boundary.
    dem_resolution : tuple
        DEM resolution (x_res, y_res).
    output_path : str, optional
        If provided, save the preprocessed raster.
    
    Returns:
    --------
    preprocessed_data : numpy.ndarray
    preprocessed_meta : dict
    """
    with rasterio.open(mrlc_path) as src:
        # Ensure CRS match for initial clipping
        watershed_gdf_src_crs = watershed_gdf.to_crs(src.crs)
        watershed_geom = [geom.__geo_interface__ for geom in watershed_gdf_src_crs.geometry]
        
        # Step 1: Clip to watershed in original CRS
        clipped_data, clipped_transform = mask(src, watershed_geom, crop=True, filled=True, nodata=0)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": clipped_transform,
            "nodata": src.nodata if src.nodata is not None else 250
        })
        
        # Step 2: Reproject to watershed CRS
        target_crs = watershed_gdf.crs
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, clipped_meta['width'], clipped_meta['height'], *rasterio.transform.array_bounds(
                clipped_meta['height'], clipped_meta['width'], clipped_transform
            ), resolution=dem_resolution
        )
        
        kwargs = clipped_meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata if src.nodata is not None else 250
        })
        
        dest = np.empty((clipped_data.shape[0], height, width), dtype=clipped_data.dtype)
        
        for i in range(clipped_data.shape[0]):
            reproject(
                source=clipped_data[i],
                destination=dest[i],
                src_transform=clipped_transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
        
        # Step 3: Handle NoData (no replacement, preserve 250 as NoData)
        
        # Optional save
        if output_path:
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(dest)
        
        return dest, kwargs
    

def identify_main_streamline(profiles: List[int], connections: dict) -> List[bool]:
    """Identifies the main streamline within a river network based on segment connectivity and length.

    The main streamline is determined by finding the longest continuous path of river segments
    from a source (headwater) to the outlet of the river network. The length is calculated
    based on the number of unique cells in the segments, accounting for overlapping cells
    at segment connections.

    Args:
        profiles (list): A list where each element represents a river segment. Each segment
                         is a list of channel indices (cells) that constitute that part of the
                         river. This is typically the output of `pyshed.grid.extract_profiles`.
        connections (dict): A dictionary describing the connectivity between river segments.
                            Keys are segment indices, and values are the segment indices
                            immediately downstream. The outlet segment is identified as a
                            segment that flows into itself (e.g., `connections[segment_id] == segment_id`).
                            This is typically the output of `pyshed.grid.extract_profiles`.

    Returns:
        list: A boolean list of the same length as `profiles`. Each element is `True` if the
              corresponding river segment is part of the identified main streamline, and `False` otherwise.

    Raises:
        ValueError: If no outlet segment is found within the `connections` dictionary.
    """
    # Find the outlet segment (where segment flows into itself)
    outlet = None
    for segment, downstream in connections.items():
        if segment == downstream:
            outlet = segment
            break
    if outlet is None:
        raise ValueError("No outlet found in connections.")
    
    # Identify source segments (segments not downstream of any other segment)
    all_segments = set(connections.keys())
    downstream_segments = set(connections.values())
    sources = all_segments - downstream_segments
    
    # Memoized function to compute path length in unique cells from segment to outlet
    length_memo = {}
    
    def path_length(segment):
        if segment == outlet:
            return len(profiles[segment])
        if segment in length_memo:
            return length_memo[segment]
        downstream = connections[segment]
        # Subtract 1 to account for overlapping cell at connection
        length_memo[segment] = len(profiles[segment]) + path_length(downstream) - 1
        return length_memo[segment]
    
    # Find the source with the longest path to the outlet
    max_length = -1
    main_source = None
    for source in sources:
        length = path_length(source)
        if length > max_length:
            max_length = length
            main_source = source
    
    # Trace the main streamline from the main source to the outlet
    main_streamline_segments = []
    current = main_source
    while current != outlet:
        main_streamline_segments.append(current)
        current = connections[current]
    main_streamline_segments.append(outlet)
    
    # Create a boolean list indicating main streamline segments
    is_main = [segment in main_streamline_segments for segment in range(len(profiles))]
    
    return is_main

def determine_threshold_river_network(area_km2, k=10, min_thresh=30, max_thresh=1000):
    """
    Determine the threshold for the accumulation raster based on watershed area.
    This function calculates accumulation threshold value, for extracting river network,
    based on the area of the watershed in square kilometers.
    """
    threshold = int(k * area_km2)

    return max(min(threshold, max_thresh), min_thresh)



# ---------- BASIC UTILS ---------- #

def load_raster(raster_path):
    """Load raster and return the dataset and array"""
    with rasterio.open(raster_path) as src:
        array = src.read(1)
        profile = src.profile
    return array, profile

def get_masked_array(raster_path, geometry):
    """Mask raster using geometry and return masked array"""
    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, geometry, crop=True)
        out_image = out_image[0]
    return out_image

from pyproj import Transformer, CRS
from typing import Tuple
import pyproj


def degree_res_to_meter_res(degree_res: Tuple[float, float], lon: float, lat: float,
                            from_crs: pyproj.CRS, target_crs: pyproj.CRS) -> Tuple[float, float]:
    """
    Convert resolution from degrees to meters based on local scale at a given geographic point.

    Parameters:
    -----------
    degree_res : tuple of float
        Resolution in degrees (x_res_deg, y_res_deg).
    lon : float
        Longitude of the reference point (e.g., watershed centroid) (shoud be in same crs of source of degree_res). Usualally watershed_gdf_geometry.centroid
    lat : float
        Latitude of the reference point.
    from_crs : pyproj.CRS
        Source geographic CRS (usually EPSG:4269 for NAD83).
    target_crs : pyproj.CRS
        Target projected CRS for conversion (must be in meters), usually watershed_gdf.crs.

    Returns:
    --------
    tuple of float
        Resolution in meters (x_res_m, y_res_m).
    """
    # Set up transformer from geographic CRS (degrees) to watershed CRS (meters)
    transformer = Transformer.from_crs(from_crs, target_crs, always_xy=True)
    
    dx, dy = degree_res

    # Transform reference point
    x0, y0 = transformer.transform(lon, lat)

    # X-resolution: shift only longitude
    x1, _ = transformer.transform(lon + dx, lat)
    res_x_m = abs(x1 - x0)

    # Y-resolution: shift only latitude
    _, y2 = transformer.transform(lon, lat + dy)
    res_y_m = abs(y2 - y0)

    return res_x_m, res_y_m

# ---------- FUNCTIONS FOR EACH METRIC ---------- #


def compute_land_cover_percentages(nlcd_array: np.ndarray, return_names: bool = False) -> dict:

    """
    Calculate the percentage of each land cover type within the preprocessed NLCD array.

    Parameters:
    -----------
    nlcd_array : np.ndarray
        Preprocessed NLCD array clipped to watershed.
    return_names : bool, default=False
        If True, return class names as keys; otherwise, return class codes.

    Returns:
    --------
    dict
        Dictionary where keys are either NLCD land cover class codes or names, and values are percentages.
    """

    # NLCD class mapping
    NLCD_CLASSES = {
        250: 'Unknown',
        11: 'Open Water',
        12: 'Perennial Ice/Snow',
        21: 'Developed, Open Space',
        22: 'Developed, Low Intensity',
        23: 'Developed, Medium Intensity',
        24: 'Developed, High Intensity',
        31: 'Barren Land (Rock/Sand/Clay)',
        41: 'Deciduous Forest',
        42: 'Evergreen Forest',
        43: 'Mixed Forest',
        52: 'Shrub/Scrub',
        71: 'Grassland/Herbaceous',
        81: 'Pasture/Hay',
        82: 'Cultivated Crops',
        90: 'Woody Wetlands',
        95: 'Emergent Herbaceous Wetlands'
    }

    # Maks out out of clipped watershed boundary geometry
    valid_mask = nlcd_array != 0
    valid_data = nlcd_array[valid_mask]

    total_pixels = valid_data.size

    unique, counts = np.unique(valid_data, return_counts=True)

    percentages = {}

    for class_code, count in zip(unique, counts):
        key = NLCD_CLASSES.get(class_code, f'Unknown') if return_names else f'lc_code_{int(class_code)}'
        percentages[key] = (count / total_pixels) * 100

    return percentages

def compute_area(boundary_gpkg):
    """
    Compute the total area of the watershed.

    Parameters:
    boundary_gpkg (str): Path to the GPKG file of watershed boundary.

    Returns:
    float: Area in square meters.
    """
    gdf = gpd.read_file(boundary_gpkg)
    gdf = gdf.to_crs(epsg=3395)  # Project to metric system
    return gdf.area.sum()

def compute_drainage_density(stream_length, area):
    """
    Compute Drainage Density.

    Parameters:
    stream_length (float): Total length of stream channels (m).
    area (float): Total area of the watershed (m²).

    Returns:
    float: Drainage density (m⁻¹).
    """
    return stream_length / area

def compute_stream_length(stream_gdf):
    """
    Compute length of the main stream channel.

    Parameters:
    stream_gdf (gpd.GeoDataFrame): GeoDataFrame of stream lines.

    Returns:
    float: Length of main channel (m).
    """
    stream_gdf = stream_gdf.to_crs(epsg=3395)
    return stream_gdf.length.max()

def compute_basin_length(flow_direction_raster, mask_raster):
    """
    Compute basin length (longest flow path).

    Parameters:
    flow_direction_raster (str): Path to flow direction raster.
    mask_raster (str): Path to mask raster.

    Returns:
    float: Basin length in meters.
    """
    # Placeholder: actual implementation requires D8 flow tracing
    # Use pysheds or tauDEM to find longest flow path
    return NotImplemented

def compute_centroidal_flowpath(boundary_gpkg, outlet_point):
    """
    Compute centroidal flowpath from outlet to centroid.

    Parameters:
    boundary_gpkg (str): Path to boundary GPKG.
    outlet_point (tuple): Outlet point (x, y).

    Returns:
    float: Flowpath length (m).
    """
    gdf = gpd.read_file(boundary_gpkg)
    gdf = gdf.to_crs(epsg=3395)
    centroid = gdf.centroid.unary_union
    return centroid.distance(gpd.GeoSeries([outlet_point], crs="EPSG:3395")[0])

def compute_10_85_flowpath(basin_length):
    """
    Compute the 10-85% flowpath length.

    Parameters:
    basin_length (float): Basin length (m).

    Returns:
    float: Flowpath length from 10% to 85% of basin length.
    """
    return (0.85 - 0.10) * basin_length

def compute_main_channel_slope(elev_outlet, elev_source, stream_length):
    """
    Compute slope of main stream channel.

    Parameters:
    elev_outlet (float): Elevation at outlet (m).
    elev_source (float): Elevation at stream source (m).
    stream_length (float): Main channel length (m).

    Returns:
    float: Main channel slope.
    """
    return (elev_source - elev_outlet) / stream_length

def compute_10_85_slope(elev_10, elev_85, L_10_85):
    """
    Compute 10-85 slope.

    Parameters:
    elev_10 (float): Elevation at 10% length.
    elev_85 (float): Elevation at 85% length.
    L_10_85 (float): Length of 10-85 flowpath (m).

    Returns:
    float: 10-85 slope.
    """
    return (elev_10 - elev_85) / L_10_85

def compute_basin_slope(dem_array):
    """
    Compute average basin slope.

    Parameters:
    dem_array (np.ndarray): Elevation array.

    Returns:
    float: Average slope in radians.
    """
    from scipy.ndimage import sobel
    dx = sobel(dem_array, axis=0)
    dy = sobel(dem_array, axis=1)
    slope = np.sqrt(dx**2 + dy**2)
    return np.nanmean(slope)

def compute_basin_relief(dem_array):
    """
    Compute basin relief.

    Parameters:
    dem_array (np.ndarray): Elevation array.

    Returns:
    float: Basin relief (m).
    """
    return np.nanmax(dem_array) - np.nanmin(dem_array)

def compute_compactness_coefficient(perimeter, area):
    """
    Compute compactness coefficient.

    Parameters:
    perimeter (float): Perimeter of watershed (m).
    area (float): Area of watershed (m²).

    Returns:
    float: Compactness coefficient.
    """
    return perimeter / (2 * math.sqrt(math.pi * area))

def compute_form_factor(area, basin_length):
    """
    Compute form factor.

    Parameters:
    area (float): Watershed area (m²).
    basin_length (float): Basin length (m).

    Returns:
    float: Form factor.
    """
    return area / (basin_length ** 2)

def compute_elongation_ratio(area, basin_length):
    """
    Compute elongation ratio.

    Parameters:
    area (float): Watershed area (m²).
    basin_length (float): Basin length (m).

    Returns:
    float: Elongation ratio.
    """
    diameter = 2 * math.sqrt(area / math.pi)
    return diameter / basin_length

def compute_relief_ratio(basin_relief, basin_length):
    """
    Compute relief ratio.

    Parameters:
    basin_relief (float): Basin relief (m).
    basin_length (float): Basin length (m).

    Returns:
    float: Relief ratio.
    """
    return basin_relief / basin_length

def compute_ruggedness_number(basin_relief, drainage_density):
    """
    Compute ruggedness number.

    Parameters:
    basin_relief (float): Basin relief (m).
    drainage_density (float): Drainage density (m⁻¹).

    Returns:
    float: Ruggedness number.
    """
    return basin_relief * drainage_density

def compute_overland_flow_length(drainage_density):
    """
    Compute overland flow length.

    Parameters:
    drainage_density (float): Drainage density (m⁻¹).

    Returns:
    float: Overland flow length (m).
    """
    return 1 / (2 * drainage_density)

def compute_channel_sinuosity(main_channel_length, straight_distance):
    """
    Compute channel sinuosity.

    Parameters:
    main_channel_length (float): Main channel length (m).
    straight_distance (float): Straight-line distance (m).

    Returns:
    float: Channel sinuosity.
    """
    return main_channel_length / straight_distance

def extract_land_cover(landcover_raster, boundary_gpkg):
    """
    Extract land cover classes within watershed boundary.

    Parameters:
    landcover_raster (str): Path to land cover raster.
    boundary_gpkg (str): Path to boundary GPKG.

    Returns:
    np.ndarray: Array of land cover classes within watershed.
    """
    gdf = gpd.read_file(boundary_gpkg)
    geometries = [shape(geom) for geom in gdf.geometry]
    masked = get_masked_array(landcover_raster, geometries)
    return masked[masked != 0]

