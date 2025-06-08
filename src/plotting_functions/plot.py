import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
import geopandas as gpd
import numpy as np

def hydrograph1(df, streamflow_col, time_col, streamflow2_col=None, precip_col=None, 
               P_units="", S_units="", S1_col="#D62728", S2_col="#1F77B4", 
               stream_label="Streamflow", precip2_col=None):
    """
    Generate hydrograph and hyetograph plot.

    Parameters:
        - df: pandas DataFrame with time and data columns
        - streamflow_col: column name for primary streamflow
        - time_col: column name for time series (datetime)
        - streamflow2_col: optional second streamflow column
        - precip_col: optional precipitation column
        - precip2_col: optional second precipitation column
        - P_units: units for precipitation (string)
        - S_units: units for streamflow (string)
        - S1_col: color for primary streamflow line
        - S2_col: color for secondary streamflow line (if provided)
        - stream_label: label for streamflow y-axis
    """

    time_series = df[time_col]
    streamflow = df[streamflow_col]
    streamflow2 = df[streamflow2_col] if streamflow2_col else None
    precip = df[precip_col] if precip_col else None
    precip2 = df[precip2_col] if precip2_col else None
   
    
    # Create figure and primary axis for streamflow
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot precipitation on a twin axis (inverted)
    if precip is not None:
        ax2 = ax1.twinx()

        # Compute bar width in days (for datetime x-axis)
        if len(time_series) > 1:
            delta = (time_series.iloc[1] - time_series.iloc[0]).total_seconds() / (3600 * 24)
            bar_width = delta * 0.8
        else:
            bar_width = 0.01
            
        # Draw precipitation bars behind the streamflow line
        ax2.bar(
            time_series,
            precip,
            width=bar_width,
            color="skyblue",
            edgecolor="navy",
            linewidth=0.5,
            alpha=0.8,
            align="center",
            zorder=1,
        )
        if precip2 is not None:
            ax2.bar(
                time_series,
                precip2,
                width=bar_width,
                color="lightcoral",
                edgecolor="darkred",
                linewidth=0.5,
                alpha=0.6,
                align="center",
                zorder=1,
            )
        # Set limits to invert the precipitation axis
        max_precip = np.nanmax(precip, precip2)
        
        # Custom ticks for precipitation
        r = max(np.ceil(-np.log10(max_precip)), 0)
        tick_max = np.round(max_precip + 10**(-r), int(r))
        num_ticks = 5 if tick_max >= 10 * 10*(-r) else int(max_precip * 10*r + 1) + 1
        ax2.set_yticks(np.linspace(0, tick_max, num_ticks))
        ax2.set_ylabel(f"Precipitation ({P_units})" if P_units else "Precipitation")

        if True:
            # Add legend for precipitation
            precip_label = f"Precipitation ({P_units})" if P_units else "Precipitation"
            precip2_label = f"Precipitation 2 ({P_units})" if P_units else "Precipitation 2"
            handles = [plt.Rectangle((0,0),1,1, facecolor="skyblue", edgecolor="navy", alpha=0.8)]
            labels = [precip_label]
            if precip2 is not None:
                handles.append(plt.Rectangle((0,0),1,1, facecolor="lightcoral", edgecolor="darkred", alpha=0.6))
                labels.append(precip2_label)
            ax2.legend(handles, labels, loc="upper left", frameon=False, fontsize=10, bbox_to_anchor=(0, 0.88))
        

        # Plot primary streamflow
        ax1.plot(
            time_series,
            streamflow,
            color=S1_col,
            linewidth=2,
            label=stream_label,
            zorder=3,
        )
        # Optional second streamflow
        if streamflow2 is not None:
            ax1.plot(
                time_series,
                streamflow2,
                color=S2_col,
                linestyle="--",
                linewidth=2,
                label=f"{stream_label} 2",
                zorder=3,
            )

    
    # Configure streamflow y-axis limits with some padding
    all_streams = [streamflow]
    if streamflow2 is not None:
        all_streams.append(streamflow2)
    max_stream = np.nanmax(np.concatenate(all_streams))
    ax1.set_ylim(0, max_stream/0.75)

    # Configure inverted ppt limit axis with some padding
    ppt_tick_max = ax1.get_ylim()[1] * max_precip / (ax1.get_ylim()[1] - max_stream)
    ax2.set_ylim(1.1 * ppt_tick_max, 0) 

    # Y-axis label for streamflow
    if S_units in ["m3/s", "m3s"]:
        ax1.set_ylabel(f"{stream_label} (m³/s)", fontsize=11)
    elif S_units in ["ft3/s", "ft3s"]:
        ax1.set_ylabel(f"{stream_label} (ft³/s)", fontsize=11)
    elif S_units:
        ax1.set_ylabel(f"{stream_label} ({S_units})", fontsize=11)
    else:
        ax1.set_ylabel(stream_label, fontsize=11)
    ax1.tick_params(axis="y", labelsize=10)
    
    # X-axis formatting
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax1.set_xlabel("Date", fontsize=11)

    # Legend for streamflow lines
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # Tight layout to prevent clipping
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

def hydrograph2(df, streamflow_col, time_col, streamflow2_col=None, precip_col=None, 
                P_units="", S_units="", S1_col="#D62728", S2_col="#1F77B4", 
                stream_label="Streamflow", precip2_col=None, ppt_legend=False):
    """
    Generate hydrograph and hyetograph plot with two subplots to prevent overlap.

    Parameters:
        - df: pandas DataFrame with time and data columns
        - streamflow_col: column name for primary streamflow
        - time_col: column name for time series (datetime)
        - streamflow2_col: optional second streamflow column
        - precip_col: optional precipitation column
        - precip2_col: optional second precipitation column
        - P_units: units for precipitation (string)
        - S_units: units for streamflow (string)
        - S1_col: color for primary streamflow line
        - S2_col: color for secondary streamflow line (if provided)
        - stream_label: label for streamflow y-axis
    """

    time_series = df[time_col]
    streamflow = df[streamflow_col]
    streamflow2 = df[streamflow2_col] if streamflow2_col else None
    precip = df[precip_col] if precip_col else None
    precip2 = df[precip2_col] if precip2_col else None

    # Create figure with two subplots, sharing x-axis, no vertical space
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0}, figsize=(10, 6))

    # Plot precipitation on the top subplot (ax2)
    if precip is not None:
        # Compute bar width in days (for datetime x-axis)
        if len(time_series) > 1:
            delta = (time_series.iloc[1] - time_series.iloc[0]).total_seconds() / (3600 * 24)
            bar_width = delta * 0.8
        else:
            bar_width = 0.01

        # Determine the maximum precipitation for y-limits
        if precip2 is not None:
            max_precip = max(max_precip, np.nanmax(precip2))
        else:
            max_precip = np.nanmax(precip)

        # Plot precipitation bars
        ax2.bar(
            time_series,
            precip,
            width=bar_width,
            color="skyblue",
            edgecolor="navy",
            linewidth=0.5,
            alpha=0.8,
            align="center"
        )
        if precip2 is not None:
            ax2.bar(
                time_series,
                precip2,
                width=bar_width,
                color="lightcoral",
                edgecolor="darkred",
                linewidth=0.5,
                alpha=0.6,
                align="center"
            )

        # Invert y-axis so bars grow downward from the top
        ax2.set_ylim(max_precip , 0)
        # Move ticks and label to the right
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel(f"Precipitation ({P_units})" if P_units else "Precipitation", fontsize=11)
        ax2.tick_params(axis="y", labelsize=10)
        # Remove bottom spine and ticks
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if ppt_legend:
            # Add legend for precipitation
            precip_label = f"Precipitation ({P_units})" if P_units else "Precipitation"
            precip2_label = f"Precipitation 2 ({P_units})" if P_units else "Precipitation 2"
            handles = [plt.Rectangle((0,0),1,1, facecolor="skyblue", edgecolor="navy", alpha=0.8)]
            labels = [precip_label]
            if precip2 is not None:
                handles.append(plt.Rectangle((0,0),1,1, facecolor="lightcoral", edgecolor="darkred", alpha=0.6))
                labels.append(precip2_label)
            ax2.legend(handles, labels, loc="lower right", frameon=False, fontsize=10)

    # Plot streamflow on the bottom subplot (ax1)
    ax1.plot(
        time_series,
        streamflow,
        color=S1_col,
        linewidth=2,
        label=stream_label
    )
    if streamflow2 is not None:
        ax1.plot(
            time_series,
            streamflow2,
            color=S2_col,
            linestyle="--",
            linewidth=2,
            label=f"{stream_label} 2"
        )

    # Set y-limits for streamflow with padding
    all_streams = [streamflow]
    if streamflow2 is not None:
        all_streams.append(streamflow2)
    max_stream = np.nanmax(np.concatenate(all_streams))
    ax1.set_ylim(0, max_stream)
    ax1.set_ylabel(f"{stream_label} ({S_units})" if S_units else stream_label, fontsize=11)
    ax1.tick_params(axis="y", labelsize=10)
    # Remove top spine
    ax1.spines['top'].set_visible(False)

    # X-axis formatting on the bottom subplot
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax1.set_xlabel("Date", fontsize=11)

    # Legend for streamflow lines
    if ppt_legend:
        stream_legend_loc = 'upper right'
    else:
        stream_legend_loc = 'lower right'
    ax1.legend(loc=stream_legend_loc, frameon=False, fontsize=10)

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.show()


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
