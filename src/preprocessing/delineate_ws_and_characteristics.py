from pyproj import Transformer
import rasterio
from pysheds.view import Raster, ViewFinder
import numpy as np
from scipy.ndimage import generic_filter

def get_raster_from_vrt(
    coord: tuple[float, float],
    vrt_path: str,
    buffer_km: float = 50,
    coord_datum: str = "epsg:4269",  # NAD83
    fill_nodata: bool = True
) -> Raster:
    """
    Extracts a DEM subset from a VRT file given a coordinate and asymmetric buffer distance, and optionally fills NoData gaps.

    Parameters
    ----------
        coord (tuple[float, float]): Coordinate (lon, lat) in the specified `coord_datum` CRS.
        vrt_path (str): Path to the .vrt DEM file.
        buffer_km (float): Buffer for top, left, and right. Bottom buffer is fixed at 0.1 km.
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
            buffer_left = buffer_right = buffer_top = buffer_km * 1000  # meters
            buffer_bottom = 0.1 * 1000  # 0.1 km
        else:
            km_to_deg = 1 / 111  # ~0.009 degrees per km
            buffer_left = buffer_right = buffer_top = buffer_km * km_to_deg
            buffer_bottom = 0.1 * km_to_deg

        # Define asymmetric bounding box
        bounds = [x - buffer_left, y - buffer_bottom, x + buffer_right, y + buffer_top]
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

def derive_watershed_characteristics(lat, lon, vrt_path, buffer_km):
    """
    Delineate a watershed and derive its geomorphological characteristics using PySheds.

    Parameters:
    - lat (float): Latitude of the stream gauge.
    - lon (float): Longitude of the stream gauge.
    - vrt_path (str): Path to the VRT file containing the DEM.
    - buffer_km (float): Buffer distance in kilometers around the gauge.

    Returns:
    - dict: Dictionary of watershed characteristics.
    """
    # Open the VRT file
    with rasterio.open(vrt_path) as src:
        dem_crs = src.crs
        
        # Transform coordinates to DEM's CRS
        transformer = Transformer.from_crs("epsg:4326", dem_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        
        # Calculate buffer in meters
        buffer_m = buffer_km * 1000
        
        # Define bounding box (left, bottom, right, top)
        bounds = [x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m]
        
        # Get window from bounds
        window = src.window(*bounds)
        
        # Read DEM data
        dem = src.read(1, window=window)
        affine = src.window_transform(window)
        
        # Create PySheds Grid
        grid = Grid()
        grid.add_gridded_data(data=dem, data_name='dem', affine=affine, crs=dem_crs)
        
        # Compute flow direction and accumulation
        fdir = grid.flowdir('dem')
        acc = grid.accumulation(fdir)
        
        # Snap pour point to nearest stream cell
        col, row = ~affine * (x, y)
        row, col = int(row), int(col)
        stream_mask = acc > 100  # Threshold may need adjustment
        snapped_row, snapped_col = grid.snap_to_mask(stream_mask, (row, col))
        
        # Delineate watershed
        catchment = grid.catchment(x=snapped_col, y=snapped_row, fdir=fdir, xytype='index')
        
        # Compute characteristics
        cell_area = grid.res[0] * grid.res[1]  # m²
        
        # Area (A) in km²
        A = np.sum(catchment) * cell_area / 1e6
        
        # Stream network and drainage density (D_d)
        streams = grid.extract_river_network(fdir, stream_mask)
        stream_geoms = [shape(feature['geometry']) for feature in streams]
        stream_gdf = gpd.GeoDataFrame(geometry=stream_geoms, crs=dem_crs)
        total_stream_length = stream_gdf.length.sum() / 1000  # km
        D_d = total_stream_length / A
        
        # Basin length (L_b) in km
        flow_distance = grid.flow_distance(x=snapped_col, y=snapped_row, fdir=fdir)
        L_b = np.max(flow_distance[catchment]) / 1000
        
        # Identify main channel (longest path along stream network)
        stream_cells = (acc > 100) & catchment
        max_dist_idx = np.argmax(flow_distance * stream_cells)
        row_max, col_max = np.unravel_index(max_dist_idx, flow_distance.shape)
        L_mc = flow_distance[row_max, col_max] / 1000
        
        # Centroidal flowpath (L_ca)
        shapes = list(rasterio.features.shapes(catchment.astype(np.uint8), transform=affine))
        watershed_geom = next(shape(geom) for geom, val in shapes if val == 1)
        centroid = watershed_geom.centroid
        col_c, row_c = ~affine * (centroid.x, centroid.y)
        L_ca = flow_distance[int(row_c), int(col_c)] / 1000 if catchment[int(row_c), int(col_c)] else L_b / 2
        
        # 10-85 Flowpath (L_10-85)
        L_10_85 = L_b * (0.85 - 0.10)
        
        # Slopes
        elev_diff_mc = dem[row_max, col_max] - dem[snapped_row, snapped_col]
        S_mc = elev_diff_mc / (L_mc * 1000) if L_mc > 0 else 0
        S_10_85 = elev_diff_mc / (L_b * 1000) * (0.85 - 0.10) if L_b > 0 else 0
        dz_dx, dz_dy = np.gradient(dem, grid.res[0], grid.res[1])
        slope = np.sqrt(dz_dx**2 + dz_dy**2)
        S_b = np.mean(slope[catchment])
        
        # Basin relief (H)
        H = np.max(dem[catchment]) - np.min(dem[catchment])
        
        # Shape factors
        perimeter = watershed_geom.length / 1000  # km
        C_c = perimeter / (2 * np.sqrt(np.pi * A))
        R_f = A / (L_b ** 2)
        R_e = (2 * np.sqrt(A / np.pi)) / L_b
        R_h = H / (L_b * 1000)
        R_n = H * D_d
        
        # Overland flow length (L_of)
        L_of = 1 / (2 * D_d) if D_d > 0 else 0
        
        # Channel sinuosity (S_i)
        straight_dist = Point(x, y).distance(Point(affine * (col_max, row_max))) / 1000
        S_i = L_mc / straight_dist if straight_dist > 0 else 1
        
        # Return characteristics
        return {
            'Area (km²)': A,
            'Drainage Density (km⁻¹)': D_d,
            'Stream Length (km)': L_mc,
            'Basin Length (km)': L_b,
            'Centroidal Flowpath (km)': L_ca,
            '10-85 Flowpath (km)': L_10_85,
            'Main Channel Slope': S_mc,
            '10-85 Slope': S_10_85,
            'Basin Slope': S_b,
            'Basin Relief (m)': H,
            'Compactness Coefficient': C_c,
            'Form Factor': R_f,
            'Elongation Ratio': R_e,
            'Relief Ratio': R_h,
            'Ruggedness Number': R_n,
            'Overland Flow Length (km)': L_of,
            'Channel Sinuosity': S_i
            # 'Land Cover': Requires MRLC data
        }

# Example usage
if __name__ == "__main__":
    lat, lon = 39.9612, -82.9988  # Columbus, OH
    vrt_path = "path/to/your/ohio_dem.vrt"
    buffer_km = 50
    chars = derive_watershed_characteristics(lat, lon, vrt_path, buffer_km)
    for key, value in chars.items():
        print(f"{key}: {value:.4f}")