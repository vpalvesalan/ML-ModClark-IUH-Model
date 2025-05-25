from pyproj import Transformer
import rasterio
from shapely import geometry, ops
import geopandas as gpd
from pysheds.view import Raster, ViewFinder
from pysheds.grid import Grid
import numpy as np
from scipy.ndimage import generic_filter


from preprocessing.get_utm_epsg import get_nad83_utm_epsg


def get_raster_from_vrt(
    coord: tuple[float, float],
    vrt_path: str,
    up_stream_buffer_km: float = 30,
    down_stream_buffer_km: float = 1,
    coord_datum: str = "epsg:4269",  # NAD83
    fill_nodata: bool = True
) -> Raster:
    """
    Extracts a DEM subset from a VRT file given a coordinate and asymmetric buffer distance, and optionally fills NoData gaps.

    Parameters
    ----------
        coord (tuple[float, float]): Coordinate (lon, lat) in the specified `coord_datum` CRS.
        vrt_path (str): Path to the .vrt DEM file.
        up_stream_buffer_km (float): Buffer for top, left, and right.
        down_stream_buffer_km (float): Buffer for bottom.
        coord_datum (str): EPSG code of the input coordinate system. Default is 'epsg:4269' (NAD83).
        fill_nodata (bool): If True, fills NoData values using neighborhood averaging.

    Returns
    -------
        Raster: PySheds Raster.
    """
    with rasterio.open(vrt_path) as src:
        dem_crs = src.crs

        # Transform coordinates from coord_datum to DEM's CRS
        if coord_datum.lower() != str(dem_crs).lower():
            transformer = Transformer.from_crs(crs_from=coord_datum, crs_to=dem_crs, always_xy=True)
            x, y = transformer.transform(*coord)
        else:
            x, y = coord

        # Determine buffers (asymmetric)
        if dem_crs.is_projected:
            buffer_left = buffer_right = buffer_top = up_stream_buffer_km * 1000  # meters
            down_stream_buffer_km = down_stream_buffer_km * 1000  # meters
        else:
            km_to_deg = 1 / 111  # ~0.009 degrees per km
            buffer_left = buffer_right = buffer_top = up_stream_buffer_km * km_to_deg
            down_stream_buffer_km = down_stream_buffer_km * km_to_deg

        # Define asymmetric bounding box
        bounds = [x - buffer_left, y - down_stream_buffer_km, x + buffer_right, y + buffer_top]
        window = src.window(*bounds)

        array = src.read(1, window=window)
        transform = src.window_transform(window)
        shape = array.shape
        crs = src.crs
        nodata = src.nodata

        # Fill NoData values if requested
        if fill_nodata and nodata is not None:
            mask = array == nodata
            array = array.astype('float32')
            array[mask] = np.nan  # Replace NoData with NaN

            def mean_filter(values):
                center = values[len(values) // 2]
                if np.isnan(center):
                    return np.nanmean(values)
                return center

            filled = generic_filter(array, mean_filter, size=3, mode='nearest')
            array = np.where(np.isnan(array), filled, array)
            array = np.where(np.isnan(array), nodata, array)  # Reassign still-NaNs to nodata

        # Build viewfinder and return PySheds Raster
        view_finder = ViewFinder(
            affine=transform,
            shape=shape,
            crs=crs,
            nodata=nodata
        )

        raster = Raster(array, viewfinder=view_finder)
        return raster
    

def delineate_watershed_d8(
    raster: Raster,
    acc_threshold: float,
    coord: tuple[float, float]
) -> tuple[np.ndarray, Grid, tuple[int, int], np.ndarray]:
    """
    Delineate a watershed catchment using the D8 flow‐direction algorithm.

    This function takes a DEM raster, computes D8 flow directions
    and accumulation, snaps the user‐provided pour point to a cell
    exceeding the given accumulation threshold, delineates the
    upstream catchment, clips the grid to that catchment, and
    returns the mask of the catchment along with ancillary data.

    Parameters
    ----------
    raster : Raster
        A PySheds Raster object representing the DEM.
    acc_threshold : float
        Minimum upstream accumulation value to snap the pour point.
    coord : tuple[float, float]
        Initial pour point coordinate (x, y) in the raster CRS.

    Returns
    -------
    clipped_catch : numpy.ndarray
        Boolean mask of the delineated catchment.
    grid : Grid
        The PySheds Grid object (now clipped to the catchment).
    snapped_coord : tuple[int, int]
        The pour point snapped to the nearest high‐accumulation cell.
    fdir : numpy.ndarray
        The D8 flow‐direction grid.

    Raises
    ------
    ValueError
        If `acc_threshold` is not positive.
    """
    if acc_threshold <= 0:
        raise ValueError("`acc_threshold` must be positive.")

    # D8 direction mapping for PySheds
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Build flow direction and accumulation grids
    grid = Grid.from_raster(raster)
    fdir = grid.flowdir(raster)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # Snap pour point to cell exceeding accumulation threshold
    x_snap, y_snap = grid.snap_to_mask(mask=acc > acc_threshold, xy=coord)
    snapped_coord = (x_snap, y_snap)

    # Delineate the upstream catchment
    catch = grid.catchment(
        x=x_snap,
        y=y_snap,
        fdir=fdir,
        dirmap=dirmap,
        xytype='coordinate'
    )

    # Clip the grid to the catchment and return its mask
    grid.clip_to(catch)
    clipped_catch = grid.view(catch)

    return clipped_catch, grid, snapped_coord, fdir

def catchment_mask_to_gdf(
    grid: Grid,
    project: bool = True,
    print_log: bool = False
) -> gpd.GeoDataFrame:
    """
    Convert the PySheds Grid catchment mask to a vector GeoDataFrame
    in the appropriate NAD83 UTM zone.

    This function:
    1. Extracts polygon geometries from the raster mask in the Grid.
    2. Builds a GeoDataFrame in the original grid CRS.
    3. Determines the best NAD83 UTM EPSG code.
    4. Reprojects the GeoDataFrame to that UTM CRS.

    Parameters
    ----------
    grid : Grid
        PySheds Grid object with the catchment mask loaded and configured.
    project : bool, default True
        If True, reproject the catchment polygon to the appropriate UTM zone.
    print_log : bool, default True
        If True, print status messages.

    Returns
    -------
    catchment_gdf : geopandas.GeoDataFrame
        The catchment polygon(s) reprojected to NAD83 UTM.

    Raises
    ------
    ValueError
        If no catchment polygon can be extracted.
    """
    shapes = list(grid.polygonize())
    if not shapes:
        raise ValueError("No catchment polygon could be extracted from the grid.")

    polygons = [geometry.shape(shape) for shape, value in shapes if value]
    catchment_polygon = ops.unary_union(polygons)

    catchment_gdf = gpd.GeoDataFrame(geometry=[catchment_polygon], crs=grid.crs)

    if project:
        utm_epsg = get_nad83_utm_epsg(catchment_gdf, print_log=print_log)
        projection = f"EPSG:{utm_epsg}"

        if print_log:
            print(f"Reprojecting to {projection}...")

        catchment_gdf = catchment_gdf.to_crs(projection)

    return catchment_gdf


def save_raster(filename, array, transform, crs, dtype='int32'):
    """
    Save a NumPy array as a GeoTIFF raster file.

    Parameters
    ----------
    filename : str
        Path to the output GeoTIFF file.
    array : numpy.ndarray
        2D array representing the raster data to be saved.
    transform : affine.Affine
        Affine transformation mapping pixel coordinates to spatial coordinates.
    crs : dict or str
        Coordinate Reference System of the raster, in a format supported by rasterio.
    dtype : str, optional
        Data type of the raster to be saved. Default is 'int32'.

    Notes
    -----
    - Suitable for saving geospatial raster data like flow direction or catchment masks.
    - Ensure the array shape matches the intended spatial extent and resolution.
    """
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform, 
        nodata=array.nodata
    ) as dst:
        dst.write(array, 1)

