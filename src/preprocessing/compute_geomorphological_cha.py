import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

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
