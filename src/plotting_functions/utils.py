import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import numpy as np

def plot_watershed_river_layers(watershed_gdf, river_gdf, outlet_point, upstream_point, ridge_point=None):
    """
    Plots a watershed with river layers and key points (outlet, upstream, and optionally ridge).

    Parameters
    ----------
    watershed_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the watershed boundary geometry.
    
    river_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing river geometries. Must include a 'Type' column
        with values 'Mainstream' and/or 'Tributary'.
    
    outlet_point : tuple
        Coordinates (x, y) of the outlet point.
    
    upstream_point : tuple
        Coordinates (x, y) of a representative upstream point.
    
    ridge_point : tuple, optional
        Coordinates (x, y) of a ridge point (e.g., divide/high elevation).
        If None, the ridge point is not plotted.

    Raises
    ------
    ValueError
        If inputs are invalid, missing, or empty.
    
    Notes
    -----
    - Uses consistent CRS across all layers.
    - Filters out invalid or empty geometries.
    - Dynamically adjusts plot aspect ratio based on latitude.
    """

    # Validate GeoDataFrames
    if not isinstance(watershed_gdf, gpd.GeoDataFrame) or not isinstance(river_gdf, gpd.GeoDataFrame):
        raise ValueError("Both watershed_gdf and river_gdf must be GeoDataFrames.")
    
    if watershed_gdf.empty or river_gdf.empty:
        raise ValueError("One or both GeoDataFrames are empty.")
    
    for pt in [outlet_point, upstream_point]:
        if not (isinstance(pt, tuple) and len(pt) == 2 and all(np.isfinite(pt))):
            raise ValueError("outlet_point and upstream_point must be valid (x, y) tuples.")
    if ridge_point is not None:
        if not (isinstance(ridge_point, tuple) and len(ridge_point) == 2 and all(np.isfinite(ridge_point))):
            raise ValueError("ridge_point must be a valid (x, y) tuple if provided.")

    # Remove invalid geometries
    watershed_gdf = watershed_gdf[watershed_gdf.geometry.is_valid & ~watershed_gdf.geometry.is_empty]
    river_gdf = river_gdf[river_gdf.geometry.is_valid & ~river_gdf.geometry.is_empty]
    if watershed_gdf.empty or river_gdf.empty:
        raise ValueError("No valid geometries remain after filtering.")

    # Align CRS
    if watershed_gdf.crs != river_gdf.crs:
        river_gdf = river_gdf.to_crs(watershed_gdf.crs)

    # Start plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot watershed boundary (no fill)
    watershed_gdf.plot(ax=ax, facecolor='none', edgecolor='#1F78B4', linewidth=1, label='Watershed Boundary')

    # Plot rivers with coherent colors
    river_types = {
        'Mainstream': {'color': '#2c7fb8', 'linewidth': 2},
        'Tributary': {'color': '#a6bddb', 'linewidth': 1}
    }
    for r_type, style in river_types.items():
        r_sub = river_gdf[river_gdf['Type'] == r_type]
        if not r_sub.empty:
            r_sub.plot(ax=ax, color=style['color'], linewidth=style['linewidth'], label=r_type)
        else:
            print(f"Warning: No '{r_type}' features found.")

    # Plot key points with transparency
    ax.scatter(*outlet_point, color='#d73027', marker='o', s=100, alpha=0.65, label='Outlet Point')
    ax.scatter(*upstream_point, color='#1a9850', marker='^', s=100, alpha=0.65, label='Upstream Point')
    if ridge_point is not None:
        ax.scatter(*ridge_point, color='#984ea3', marker='s', s=100, alpha=0.65, label='Ridge Point')

    # Set aspect ratio based on mean latitude
    try:
        _, ymin, _, ymax = watershed_gdf.total_bounds
        mean_lat = np.mean([ymin, ymax])
        aspect = 1 / np.cos(np.radians(mean_lat))
        ax.set_aspect(aspect if np.isfinite(aspect) and aspect > 0 else 'equal')
    except Exception as e:
        print(f"Warning: Could not set geographic aspect ratio. Using 'equal'. Error: {e}")
        ax.set_aspect('equal')

    # Final touches
    ax.set_title('Watershed and River Network with Key Points', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt

def plot_streamlines_with_ids(gdf):
    """
    Plot streamlines from a GeoPandas DataFrame and annotate each segment with its ID.
    
    Parameters:
    gdf (GeoPandas GeoDataFrame): Input GeoDataFrame with 'id' and 'geometry' columns
    """
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the streamlines
    gdf.plot(ax=ax, color='blue', linewidth=2, label='Streamlines')
    
    # Annotate each streamline segment with its ID
    for idx, row in gdf.iterrows():
        # Get the representative point (centroid) of the linestring for annotation
        centroid = row['geometry'].centroid
        ax.annotate(
            text=str(row['id']),
            xy=(centroid.x, centroid.y),
            xytext=(5, 5),  # Offset for better visibility
            textcoords='offset points',
            fontsize=10,
            color='red',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    
    # Add title and labels
    ax.set_title('Streamlines with Segment IDs')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    
    # Add a legend
    ax.legend()
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
