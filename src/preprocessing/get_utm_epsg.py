import geopandas as gpd
from shapely.geometry import box

def get_nad83_utm_epsg(gdf : gpd.GeoDataFrame,  print_log = False) -> int:
    """
    Determine the NAD83 UTM EPSG code for a GeoPandas object in NAD83 (EPSG:4269).
    If the object spans multiple UTM zones, select the zone with the most overlap.
    
    Args:
    ----------
    gdf (GeoDataFrame): GeoPandas object in NAD83 (EPSG:4269)
    print_log (bool): If True, print debug information to the console

    Returns:
    ----------
    int: EPSG code for the appropriate NAD83 UTM projection
    """

    # Ensure the input is in NAD83 (EPSG:4269)
    if gdf.crs.to_epsg() != 4269:
        gdf = gdf.to_crs(epsg=4269)
        if print_log:
            print("Converted GeoDataFrame to NAD83 (EPSG:4269)")
    
    # Get the bounding box of the GeoDataFrame
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Function to calculate UTM zone from longitude
    def lon_to_utm_zone(lon):
        return int((lon + 180) / 6) + 1
    
    # Determine UTM zones for minx and maxx
    min_zone = lon_to_utm_zone(bounds[0])
    max_zone = lon_to_utm_zone(bounds[2])
    
    # If in the same zone
    if min_zone == max_zone:
        if print_log:
            print(f"Geometry falls entirely in UTM Zone {min_zone}")
        # NAD83 UTM zones: EPSG 269xx where xx is the zone number (04-19 for continental US)
        return 26900 + min_zone
    
    # If spanning multiple zones, calculate overlap with each zone
    if print_log:
        print(f"Geometry spans UTM Zones {min_zone} to {max_zone}")
    zones = range(min_zone, max_zone + 1)
    overlaps = []
    
    for zone in zones:
        # Create UTM zone polygon (approximate as 6-degree wide strip)
        zone_min_lon = (zone - 1) * 6 - 180
        zone_max_lon = zone * 6 - 180
        zone_poly = box(zone_min_lon, bounds[1], zone_max_lon, bounds[3])
        
        # Convert to GeoDataFrame for intersection
        zone_gdf = gpd.GeoDataFrame(geometry=[zone_poly], crs="EPSG:4269")
        
        # Calculate intersection area
        intersection = gpd.overlay(gdf, zone_gdf, how='intersection')
        if not intersection.empty:
            area = intersection.to_crs(epsg=3857).geometry.area.sum()  # Use Web Mercator for area calc
            overlaps.append((zone, area))
        else:
            overlaps.append((zone, 0))
    
    # Select zone with maximum overlap
    max_zone, max_area = max(overlaps, key=lambda x: x[1])
    
    if max_area == 0:
        raise ValueError("No valid UTM zone found for the geometry")
    
    if print_log:
        print(f"Selected UTM Zone {max_zone} with maximum overlap")
    
    # Return EPSG code for NAD83 UTM (26904 to 26919 for zones 4 to 19)
    return 26900 + max_zone