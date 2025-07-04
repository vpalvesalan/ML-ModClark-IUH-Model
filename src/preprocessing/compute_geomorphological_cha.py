from heapq import merge
from logging import warn, warning
from tracemalloc import start
from warnings import WarningMessage
import geopandas as gpd
from matplotlib.pylab import f
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
import re
import warnings
import pandas as pd
from pysheds.view import Raster
from pysheds.grid import Grid


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

def determine_threshold_river_network(area_km2, k=8, min_thresh=30, max_thresh=1000):
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
    lon: float
        Longitude of the reference point (e.g., watershed centroid) (shoud be in same crs of source of degree_res). Usualally watershed_gdf_geometry.centroid
    lat: float
        Latitude of the reference point.
    from_crs: pyproj.CRS
        Source geographic CRS (usually EPSG:4269 for NAD83).
    target_crs: pyproj.CRS
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

def compute_area(watershed_gdf: gpd.GeoDataFrame) -> float:
    """
    Compute the total area in square meters of the watershed.

    Parameters:
        watershed_gdf (gpd.GeoDataFrame): geopandas of watershed boundary.

    Returns:
        float: Area in square meters.
    """
    if not watershed_gdf.crs or not watershed_gdf.crs.is_projected:
        raise ValueError("Watershed boundary must be in a projected CRS to compute area in square meters.")
    
    return watershed_gdf.area.sum()

def compute_stream_length(stream_gdf: gpd.GeoDataFrame, mainstream_col: str = 'mainstream') -> Tuple[float, float]:
    """
    Compute the mainstream length and total length of stream channels in the watershed.

    Parameters:
        stream_gdf (gpd.GeoDataFrame): GeoDataFrame of stream lines.
        mainstream_col (str): Optional. Column name of boolean feature indicating mainstream channels (default is 'mainstream').

    Returns:
        Tuple[float, float]: Mainstream length and total length of stream channels in meters.
    """
    if not stream_gdf.crs or not stream_gdf.crs.is_projected:
        raise ValueError("Stream lines must be in a projected CRS to compute length in meters.")
    
    if mainstream_col not in stream_gdf.columns  or mainstream_col is None:
        warnings.warn(f"Column '{mainstream_col}' not found in stream GeoDataFrame. Assuming all streams are mainstream.")
        mainstream_col = 'mainstream'
        stream_gdf[mainstream_col] = True

    if not pd.api.types.is_bool_dtype(stream_gdf[mainstream_col]):
        warnings.warn(f"Column '{mainstream_col}' is not boolean. Assuming all streams are mainstream.")
        stream_gdf[mainstream_col] = True

    total_length = stream_gdf.length.sum()
    mainstream_length = stream_gdf[stream_gdf[mainstream_col]].length.sum()
    
    return mainstream_length, total_length


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

import rasterio
import geopandas as gpd

def compute_basin_length(acc: Raster, ridge_dist: Raster, gdf: gpd.GeoDataFrame) -> float:
    """
    Calculate the basin length using a pysheds accumulation raster and a GeoPandas linestring object.
    
    Parameters:
        - acc (Raster): pysheds accumulation raster
        - ridge_dist (Raster): pysheds ridge distance raster
        - gdf (gpd.GeoDataFrame): GeoDataFrame with linestring geometries, 'id' column for segment sequence, 
           and 'mainstream' boolean column for mainstream segments. gdf must be projected to meters.
    
    Returns:
        - float: Basin length which is defined as the sum of mainstream length and distance to ridge from the most ditant
        point in the mainstream segment.
    """

    if not gdf.crs.is_projected:
        raise ValueError("GeoDataFrame must be in a projected CRS (meters) for this calculation.")
    
    original_gdf_crs = gdf.crs
    if gdf.crs != acc.crs:
        gdf = gdf.to_crs(acc.crs) 
    mainstream_gdf = gdf[gdf['mainstream']]

    # Identify the first segment (assuming smallest 'id' is the first in sequence)
    first_segment = mainstream_gdf.loc[mainstream_gdf['id'].idxmin()]
    line = first_segment.geometry

    point_start = line.coords[0]  # Starting point (x, y)
    point_end = line.coords[-1]   # Ending point (x, y)
    
    accumulation_array = np.asarray(acc)
    
    # Helper function to get raster value at a point
    grid = Grid()
    grid.viewfinder  = acc.viewfinder
    def get_value_at_point(array, point):
        col, row = grid.nearest_cell(x=point[0], y=point[1],snap= 'corner')
        return array[row, col]
    
    accum_start = get_value_at_point(accumulation_array, point_start)
    accum_end = get_value_at_point(accumulation_array, point_end)

    point_upstream = point_start if accum_start < accum_end else point_end
    
    # Get the distance in cell numbers to ridge for the upstream point
    ridge_distance = get_value_at_point(ridge_dist, point_upstream)

    # Calculate the total mainstream length by summing lengths of all mainstream segments
    if original_gdf_crs != acc.crs:
        mainstream_gdf = mainstream_gdf.to_crs(original_gdf_crs)
    mainstream_length = mainstream_gdf['geometry'].length.sum()

    # Get distance in meters
    res_meters = degree_res_to_meter_res(
        (abs(ridge_dist.affine[0]), abs(ridge_dist.affine[4])),
        point_upstream[0],
        point_upstream[1],
        from_crs=grid.crs,
        target_crs=mainstream_gdf.crs)
    ridge_distance_meters = ridge_distance * ((res_meters[0]+res_meters[1])/2)

    # Compute basin length as the sum of mainstream length and ridge distance
    basin_length = mainstream_length + ridge_distance_meters
    
    return basin_length


import geopandas as gpd
import shapely
from shapely.ops import nearest_points
import networkx as nx
from collections import defaultdict
import warnings
import geopandas as gdp

def compute_centroidal_flowpath_with_networkx(river_network: gdp.GeoDataFrame, watershed_boundary: gdp.GeoDataFrame) -> float:
    """
    Computes the length of the centroidal flowpath within a river network using a
    graph-based approach with NetworkX.

    The centroidal flowpath is defined as the length of the mainstream river
    from the point nearest to the watershed's centroid, downstream to the
    segment with the highest 'id' (assumed to be the most downstream point
    or outlet of the mainstream network).

    This function builds a directed graph of the river network segments based on
    their connectivity (start/end points) and then uses NetworkX's shortest path
    algorithm to determine the path and calculate its length.

    Parameters:
        river_network (gdp.GeoDataFrame): A GeoDataFrame representing the river
            network. It must contain:
            - A boolean column named 'mainstream' (True for mainstream segments).
            - An integer column named 'id' for uniquely identifying segments.
              Segments are assumed to be ordered from upstream (lower ID) to
              downstream (higher ID) for pathfinding logic.
            The CRS must be projected.
        watershed_boundary (gdp.GeoDataFrame): A GeoDataFrame representing the
            watershed's boundary. Its CRS will be reprojected to match the
            river network's CRS if they differ.

    Returns:
        float: The computed length of the centroidal flowpath in the units
               of the `river_network`'s Coordinate Reference System (CRS).

    Raises:
        ValueError: If the `river_network` is not in a projected CRS.
        ValueError: If the identified centroidal point on the main river is
                    not sufficiently close to any mainstream segment (based on
                    an internal tolerance).
        ValueError: If no valid path exists from the identified most
                    downstream segment ('id' with `idxmax()`) to the target
                    segment containing the centroidal point within the constructed
                    network graph. This can occur if the network is disconnected
                    or the assumed 'start' and 'target' are not properly linked.

    Warns:
        UserWarning: If the `watershed_boundary` CRS is different from the
                     `river_network` CRS, indicating it has been reprojected.
    """
    return "Use `compute_centroidal_flowpath` instead of `compute_centroidal_flowpath_with_networkx` as it is more efficient and simpler."
    # Check CRS compatibility and projection
    if not river_network.crs.is_projected:
        raise ValueError("River network must be in a projected CRS.")
    if river_network.crs != watershed_boundary.crs:
        watershed_boundary = watershed_boundary.to_crs(river_network.crs)
        warnings.warn("Watershed boundary CRS was different from river network CRS. It has been reprojected to match.")
    
    centroid = watershed_boundary.geometry.iloc[0].centroid

    main_river = river_network[river_network['mainstream']]
    
    # Compute the unary union of mainstream river geometries
    main_river_union = main_river.geometry.unary_union
    
    # Step 2: Identify the nearest point on the main river to the centroid (centroidal point)
    centroidal_point = nearest_points(main_river_union, centroid)[0]
    
    # Identify the start segment (segment with highest 'id')
    start_segment = main_river.loc[main_river['id'].idxmax()]
    start_id = start_segment['id']
    
    # Build a directed graph to represent segment connectivity
    G = nx.DiGraph()
    G.add_nodes_from(main_river['id'])
    
    # Create dictionaries to map start and end points to segment IDs
    start_dict = defaultdict(list)
    end_dict = defaultdict(list)
    
    for idx, row in main_river.iterrows():
        seg_id = row['id']
        line = row.geometry
        first_point = shapely.geometry.Point(line.coords[0])
        last_point = shapely.geometry.Point(line.coords[-1])
        start_dict[first_point].append(seg_id)
        end_dict[last_point].append(seg_id)
    
    # Add edges where segments connect (last point of one to first point of another)
    for point, end_segments in end_dict.items():
        if point in start_dict:
            start_segments = start_dict[point]
            for end_seg in end_segments:
                for start_seg in start_segments:
                    G.add_edge(end_seg, start_seg)
    
    # Find the target segment containing the centroidal point
    tolerance = 1e-6
    target_segment = None
    min_distance = float('inf')
    for idx, row in main_river.iterrows():
        distance = row.geometry.distance(centroidal_point)
        if distance < min_distance:
            min_distance = distance
            target_segment = row
    if min_distance > tolerance:
        raise ValueError("Centroidal point is not on any mainstream segment.")
    target_id = target_segment['id']
    
    # Step 3: Compute the length from centroidal point to the start
    # Find the path from start segment to target segment
    try:
        path = nx.shortest_path(G, source=start_id, target=target_id)
    except nx.NetworkXNoPath:
        raise ValueError("No path exists from start segment to target segment.")
    
    # Get the ordered list of segments in the path
    path_segments = [main_river[main_river['id'] == seg_id].iloc[0] for seg_id in path]
    
    # Calculate total length
    if len(path_segments) == 1:
        # Single segment case: distance from start to centroidal point
        line = path_segments[0].geometry
        total_length = line.project(centroidal_point)
    else:
        # Multiple segments: sum full lengths of all but last, plus partial last segment
        total_length = sum(seg.geometry.length for seg in path_segments[:-1])
        last_line = path_segments[-1].geometry
        distance_along = last_line.project(centroidal_point)
        total_length += distance_along
    
    return total_length


import geopandas as gdp
from shapely.ops import nearest_points
import warnings

def compute_centroidal_flowpath(mainstream: gdp.GeoDataFrame, watershed_boundary: gdp.GeoDataFrame) -> float:
    """
    Computes the length of the centroidal flowpath within a river network.

    The centroidal flowpath is defined as the length of the mainstream river
    from the point nearest to the watershed's centroid, downstream to the
    outlet (segment with the highest 'id').

    Parameters:
        mainstream (gdp.GeoDataFrame): A GeoDataFrame representing the main river
            stream line. It must contain an integer column named 'id'
            for ordering segments from upstream (lower ID) to downstream (higher ID).
            The CRS must be projected.
        watershed_boundary (gdp.GeoDataFrame): A GeoDataFrame representing the
            watershed's boundary. Its CRS will be reprojected to match the
            river network's CRS if they differ.

    Returns:
        float: The computed length of the centroidal flowpath in the units
               of the `river_network`'s Coordinate Reference System (CRS).

    Raises:
        ValueError: If the `river_network` is not in a projected CRS.
        ValueError: If no valid path exists from the identified centroidal
                    point on the mainstream to the headwaters segment,
                    which typically means the centroidal point is downstream
                    of the headwaters.

    Warns:
        UserWarning: If the `watershed_boundary` CRS is different from the
                     `river_network` CRS, indicating it has been reprojected.
    """
    # Check CRS compatibility and projection
    if not mainstream.crs.is_projected:
        raise ValueError("River network must be in a projected CRS.")
    if mainstream.crs != watershed_boundary.crs:
        watershed_boundary = watershed_boundary.to_crs(mainstream.crs)
        warnings.warn("Watershed boundary CRS was different from river network CRS. It has been reprojected to match.")
    
    # Step 1: Compute the centroid of the watershed boundary
    centroid = watershed_boundary.geometry.iloc[0].centroid
    
    # Filter the mainstream river segments
    main_river = mainstream.sort_values('id')
    
    # Compute the unary union of mainstream river geometries
    main_river_union = main_river.geometry.unary_union
    
    # Step 2: Identify the nearest point on the main river to the centroid (centroidal point)
    centroidal_point = nearest_points(main_river_union, centroid)[0]
    
    # Find the target segment containing the centroidal point
    target_segment = None
    min_distance = float('inf')
    tolerance = 1e-6  # Tolerance for distance check
    for _, row in main_river.iterrows():
        distance = row.geometry.distance(centroidal_point)
        if distance < min_distance:
            min_distance = distance
            target_segment = row
    if min_distance > tolerance:
        raise ValueError("Centroidal point is not on any mainstream segment.")
    target_id = target_segment['id']
    
    # Step 3: Compute the length from centroidal point to the start
    total_length = 0.0
    reached_target = False
    
    # Iterate through segments in order (highest to lowest ID, downstream to upstream)
    main_river = main_river.sort_values('id',ascending=False)  # Ensure segments are ordered by ID
    for _, row in main_river.iterrows():
        seg_id = row['id']
        line = row.geometry
        
        # Stop if we've passed the target segment
        if reached_target:
            break
            
        # If this is the target segment, compute partial length and stop
        if seg_id == target_id:
            distance_along = line.project(centroidal_point)
            total_length += distance_along
            reached_target = True
        # Otherwise, add the full length of the segment
        else:
            total_length += line.length
    
    # If no path to target, raise error (e.g., target is downstream of start)
    if not reached_target:
        raise ValueError("No valid path exists from start segment to target segment.")
    
    return total_length

import geopandas as gpd
import shapely.ops as ops
from shapely.geometry import LineString

def compute_10_85_flowpath(mainstream):
    """
    Extracts the 10% to 85% segment of a river's mainstream based on cumulative length.

    This function takes a GeoPandas DataFrame containing LineString geometries representing 
    the segments of a river's mainstream, ordered from upstream to downstream by the 'ID' column. 
    It merges these LineStrings into a single continuous LineString and extracts the central 
    portion of the river corresponding to 10% to 85% of the total river length (measured 
    from upstream to downstream).

    Parameters:
        mainstream (geopandas.GeoDataFrame):
            A GeoDataFrame with LineString geometries and an 'id' column indicating 
            the order of river segments from upstream (lower IDs) to downstream (higher IDs).
            The CRS must be projected to ensure accurate length calculations.

    Returns:
        list [float, geopandas.GeoDataFrame]: The length of the extracted segment and,
            a new GeoDataFrame containing the extracted LineString 
            segment between 10% and 85% of the river's total length.

    Raises:
        ValueError: If the input CRS is not projected.
        ValueError: If the LineStrings cannot be merged into a single LineString.
    """
    
    if not mainstream.crs.is_projected:
        raise ValueError("The CRS must be projected for accurate length calculations.")
    
    # Sort by 'ID' to ensure upstream-to-downstream order
    mainstream = mainstream.sort_values(by='id')
    
    # Merge all LineStrings into a single LineString
    lines = mainstream.geometry.tolist()
    merged_line = ops.linemerge(lines)
    
    # Verify the merged result is a single LineString
    if not isinstance(merged_line, LineString):
        raise ValueError("The geometries could not be merged into a single LineString.")
    
    # Calculate total length
    total_length = merged_line.length
    
    # Calculate distances from the outlet (end) upstream
    # 10% from outlet = 90% from start, 85% from outlet = 15% from start
    start_dist = total_length * 0.15  # 85% from outlet
    end_dist = total_length * 0.9     # 10% from outlet
    
    # Extract the segment between these distances
    segment = ops.substring(merged_line, start_dist, end_dist)
    
    # Create a new GeoDataFrame with the segment
    new_gdf = gpd.GeoDataFrame(geometry=[segment], crs=mainstream.crs)
    
    # Compute the segment length
    segment_length = segment.length
    
    return [segment_length, new_gdf]

def compute_channel_slope(channel:  gpd.GeoDataFrame, dem : Raster) -> float:
    """
    Compute the  slope of a channel.

    Parameters:
        channel (gpd.GeoDataFrame): GeoDataFrame representing a channel.
        dem (Raster): pysheds Raster object of the DEM.

    Returns:
        float: Main channel slope.
    """
    if channel.empty:
        raise ValueError("Channel GeoDataFrame is empty.")
    if not channel.crs.is_projected:
        raise ValueError("Channel GeoDataFrame must be in a projected CRS (meters).")
    
    stream_length = channel.length.sum()
    if stream_length == 0:
        raise ValueError("Stream length is zero. Ensure the channel geometry is valid and has length.")

    if channel.to_crs != dem.crs:
        channel = channel.to_crs(dem.crs)
    
    if 'id' not in channel.columns:
        warnings.warn("Channel GeoDataFrame does not have an 'id' column. Assuming segments are ordered by geometry.")
    else:
        channel = channel.sort_values(by='id')

    line = channel.geometry.tolist()
    merged_line = ops.linemerge(line)

    end_point = merged_line.coords[-1]
    start_point = merged_line.coords[0]

    grid = Grid()
    grid.viewfinder = dem.viewfinder

    def get_value_at_point(array, point):
        col, row = grid.nearest_cell(x=point[0], y=point[1],snap= 'corner')
        return array[row, col]
    start_elev = get_value_at_point(dem, start_point)
    end_elev = get_value_at_point(dem, end_point)

    if np.isnan(start_elev) or np.isnan(end_elev):
        raise ValueError("Elevation values at start or end point are NaN. Ensure DEM is valid and has no missing data.")
    
    # Compute slope as (elevation difference) / (length of channel)
    slope = abs(start_elev - end_elev) / stream_length
    return slope
    


import numpy as np
import pyproj
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pysheds.grid import Grid

def compute_average_slope(dem, target_crs='EPSG:3857', catchment_mask=None):
    """
    Compute the average slope from a DEM Raster, optionally within a catchment mask.
    
    The DEM is reprojected to the specified projected CRS (e.g., EPSG:3857) to ensure 
    slope calculations are performed in linear units (meters per meter).
    
    Args:
        dem (Raster): A pysheds Raster object representing the elevation data.
        target_crs (str or pyproj.CRS): The target coordinate reference system to project 
            the DEM into. Must be a projected CRS (units in meters).
        catchment_mask (Raster, optional): An optional Raster object representing a mask 
            (1 = inside catchment, 0 or np.nan = outside). Will be reprojected if provided.
    
    Returns:
        float: Average slope in meters per meter (m/m), ignoring NaN and masked values.
    
    Raises:
        ValueError: If target_crs is not a projected CRS (i.e., has angular units like degrees).
        TypeError: If inputs are not valid pysheds Raster objects.
    """
    
    # --- Input validation ---
    if not isinstance(dem, Raster):
        raise TypeError("Input 'dem' must be a pysheds Raster object.")
    if catchment_mask is not None and not isinstance(catchment_mask, Raster):
        raise TypeError("Input 'catchment_mask' must be a pysheds Raster object or None.")
    
    # Parse and validate CRS
    try:
        crs_obj = pyproj.CRS.from_user_input(target_crs)
    except pyproj.exceptions.CRSError:
        raise ValueError(f"'{target_crs}' is not a valid CRS identifier.")
    
    if not crs_obj.is_projected:
        raise ValueError(f"Target CRS '{target_crs}' is not a projected CRS. Please use a CRS with linear units like meters (e.g., EPSG:3857).")

    # --- Reproject DEM ---
    dem_epsg_str = f"EPSG:{dem.crs.to_epsg()}"
    crs_obj_epsg_str = str(crs_obj)
    if dem_epsg_str != crs_obj_epsg_str:
        dem_proj = dem.to_crs(crs_obj)
    
    # Create a new grid for the reprojected DEM
    grid = Grid()
    grid.viewfinder = dem_proj.viewfinder
    
    # Compute flow direction and slope
    fdir = grid.flowdir(dem_proj)
    slope = grid.cell_slopes(dem_proj, fdir) 

    slope_arr = np.asarray(slope.view())

    # --- Apply catchment mask if provided ---
    if catchment_mask is not None:
        
        # The slope raster's CRS is dem_proj.crs
        if catchment_mask.crs != dem_proj.crs:
            mask_proj = catchment_mask.to_crs(dem_proj.crs)
        else:
            mask_proj = catchment_mask

        mask_array = np.asarray(mask_proj.view())

        # If the reprojected mask's shape still doesn't match the slope array's shape,
        # perform a final grid alignment using rasterio.reproject.
        if slope_arr.shape != mask_array.shape:
            # Create an empty array for the aligned mask
            aligned_mask = np.empty(slope_arr.shape, dtype=mask_array.dtype)
            # Reproject the mask to match the slope's grid, transform, and CRS
            reproject(
                source=mask_array,
                destination=aligned_mask,
                src_transform=mask_proj.affine,
                src_crs=mask_proj.crs,
                dst_transform=slope.affine,
                dst_crs=slope.crs,           
                resampling=Resampling.nearest
            )
            mask_array = aligned_mask

        # Mask the slope array using the reprojected catchment mask
        slope_arr = np.where(mask_array > 0, slope_arr, np.nan)
        
    # --- Compute average slope ---
    avg_slope = np.nanmean(slope_arr)

    return avg_slope


def compute_basin_relief(dem: Raster, mask=None):
    """
    Compute the basin relief from a DEM Raster, optionally within a catchment mask.

    Args:
        dem (Raster): A pysheds Raster object representing the elevation data.
        mask (Raster, optional): An optional pysheds Raster object representing a mask
    
    Returns:
        float: Basin relief in meters, calculated as the difference between the maximum and minimum elevation values.
    """

    dem_array = np.asarray(dem.view())
    # --- Apply catchment mask if provided ---
    if mask is not None:

        mask_array = np.asarray(mask.view())
        if dem_array.shape != mask_array.shape:

            # Create an empty array for the aligned mask
            aligned_mask = np.empty(dem_array.shape, dtype=mask_array.dtype)

            # Reproject the mask to match the slope's grid, transform, and CRS
            reproject(
                source=mask_array,
                destination=aligned_mask,
                src_transform=mask.affine,
                src_crs=mask.crs,
                dst_transform=dem.affine,
                dst_crs=dem.crs,           
                resampling=Resampling.nearest
            )
            mask_array = aligned_mask

        # Mask the slope array using the reprojected catchment mask
        dem_array = np.where(mask_array > 0, dem_array, np.nan)

    return np.nanmax(dem_array) - np.nanmin(dem_array)

def compute_compactness_coefficient(watershed_boundary: gpd.GeoDataFrame) -> float:
    """
    Compute compactness coefficient.

    Args:
        watershed_boundary (gpd.GeoDataFrame): GeoDataFrame representing the watershed boundary.

    Returns:
        float: Compactness coefficient as the ratio of the watershed perimeter
        to the circumference of a circle with the same area.
    
    Raises:
        ValueError: If the watershed boundary is not in a projected CRS.
    """
    if not watershed_boundary.crs.is_projected:
        raise ValueError("Watershed boundary must be in a projected CRS to compute area and perimeter in meters.")
    area = watershed_boundary.area.sum()
    perimeter = watershed_boundary.length.sum()
    return perimeter / (2 * math.sqrt(math.pi * area))

def compute_form_factor(area, basin_length):
    """
    Compute form factor.

    Args:
        area (float): Watershed area (m²).
        basin_length (float): Basin length (m).

    Returns:
        float: Form factor.
    """
    return area / (basin_length ** 2)

def compute_elongation_ratio(area, basin_length):
    """
    Compute elongation ratio.

    Args:
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

    Args:
        basin_relief (float): Basin relief (m).
        basin_length (float): Basin length (m).

    Returns:
        float: Relief ratio.
    """
    return basin_relief / basin_length

def compute_ruggedness_number(basin_relief, drainage_density):
    """
    Compute ruggedness number.

    Args:
        basin_relief (float): Basin relief (m).
        drainage_density (float): Drainage density (m⁻¹).

    Returns:
        float: Ruggedness number.
    """
    return basin_relief * drainage_density

def compute_overland_flow_length(drainage_density):
    """
    Compute overland flow length.

    Args:
        drainage_density (float): Drainage density (m⁻¹).

    Returns:
        float: Overland flow length (m).
    """
    return 1 / (2 * drainage_density)

def compute_channel_sinuosity(mainstream: gpd.GeoDataFrame) -> float:
    """
    Compute channel sinuosity.

    Args:
        mainsstream (gpd.GeoDataFrame): GeoDataFrame representing the mainstream channel.

    Returns:
        float: Channel sinuosity.
    """
    
    # Sort by 'ID' to ensure upstream-to-downstream order
    mainstream = mainstream.sort_values(by='id')
    
    # Merge all LineStrings into a single LineString
    lines = mainstream.geometry.tolist()
    merged_line = ops.linemerge(lines)
    
    # Verify the merged result is a single LineString
    if not isinstance(merged_line, LineString):
        raise ValueError("The geometries could not be merged into a single LineString.")
    # Calculate total length
    main_channel_length = merged_line.length
    if main_channel_length == 0:
        raise ValueError("Total length of the channel is zero. Ensure the channel geometry is valid and has length.")
    
    merged_line_coords = list(merged_line.coords)
    start = merged_line_coords[0]
    end = merged_line_coords[-1]
    # Calculate straight-line distance between start and end points
    straight_distance = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    
    if straight_distance == 0:
        raise ValueError("Straight-line distance is zero. Ensure the channel geometry is valid and has length.")

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

def compute_land_cover_percentages(nlcd_array: np.ndarray, return_names: bool = False) -> dict:

    """
    Calculate the percentage of each land cover type within the preprocessed NLCD array.

    Args:
    -----------
    nlcd_array (np.ndarray): 
        Preprocessed NLCD array clipped to watershed.
    return_names (bool,  default=False):
        If True, return class names as keys; otherwise, return class codes.

    Returns:
    --------
    dict:
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

    percentages = {
        NLCD_CLASSES[code] if return_names else f'lc_code_{int(code)}': np.float64()
        for code in NLCD_CLASSES
    }

    for class_code, count in zip(unique, counts):
        key = NLCD_CLASSES.get(class_code) if return_names else f'lc_code_{int(class_code)}'
        
        # Only update if the key is a recognized NLCD class
        if key in percentages: 
            percentages[key] = (count / total_pixels) * 100
        elif not return_names and f'lc_code_{int(class_code)}' in percentages: # Handle case if return_names is False and code exists
            percentages[key] = (count / total_pixels) * 100

    return percentages

