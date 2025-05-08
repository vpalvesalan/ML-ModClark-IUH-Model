import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi, ConvexHull
import math

def voronoi_finite_polygons_2d(vor, radius=None, buffer_ratio=10.0):
    """
    Compute finite Voronoi regions for a 2D Voronoi diagram.

    This function processes a 2D Voronoi diagram (as produced by scipy.spatial.Voronoi) and converts any
    infinite regions into finite regions by extending their open ridges. The extension is performed along a
    direction determined by the convex hull of the input points to maintain proper orientation. If a radius is
    not provided, it is computed based on the spatial extent of the input points and scaled by a buffer ratio.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        A Voronoi diagram object with 2-dimensional input points.
    radius : float, optional
        Distance to extend infinite ridges to close regions. If None, a default value is calculated using the
        bounding box of the input points multiplied by buffer_ratio.
    buffer_ratio : float, default=10.0
        Factor to scale the default radius based on the maximum extent (width or height) of the input points.

    Returns
    -------
    new_regions : list of list of int
        A list where each element is a list of indices representing a finite polygonal region.
    new_vertices : numpy.ndarray of shape (N, 2)
        Array of 2D coordinates corresponding to the vertices of the finite regions. New vertices are added
        for previously infinite edges.

    Raises
    ------
    ValueError
        If the input Voronoi diagram does not consist of 2-dimensional points.

    Notes
    -----
    Infinite regions are identified by negative vertex indices. For each such region, the function finds the
    corresponding ridge, computes a direction based on the unit normal and the convex hull center, and extends
    the ridge by a specified radius to generate a finite vertex. The vertices for each region are then sorted in
    counterclockwise order using the centroid of the region's vertices.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    
    # Calculate convex hull of input points for directionality
    hull = ConvexHull(vor.points)
    hull_points = vor.points[hull.vertices]
    hull_center = np.mean(hull_points, axis=0)
    
    if radius is None:
        min_x, max_x = np.min(vor.points[:,0]), np.max(vor.points[:,0])
        min_y, max_y = np.min(vor.points[:,1]), np.max(vor.points[:,1])
        radius = buffer_ratio * max(max_x - min_x, max_y - min_y)

    ridge_dict = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_dict.setdefault(p1, []).append((p2, v1, v2))
        ridge_dict.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        finite_verts = [v for v in vertices if v >= 0]
        ridges = ridge_dict.get(p1, [])

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = 0.5 * (vor.points[p1] + vor.points[p2])
            direction = np.sign(np.dot(midpoint - hull_center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            finite_verts.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        if finite_verts:
            vs = np.array([new_vertices[v] for v in finite_verts])
            c = vs.mean(axis=0)  # Use region's vertex centroid for sorting
            angles = np.arctan2(vs[:,1]-c[1], vs[:,0]-c[0])
            finite_verts = np.array(finite_verts)[np.argsort(angles)]
            new_regions.append(finite_verts.tolist())

    return new_regions, np.array(new_vertices)

def derive_thiessen_polygons(gdf_points, clipping_gdf, preserve_attribute):
    """
    Generate robust Thiessen (Voronoi) polygons from point data clipped to an irregular boundary.

    This function derives Thiessen polygons from a set of input points provided in a GeoDataFrame.
    It computes a Voronoi diagram, converts any infinite regions into finite polygons using an adaptive
    radius (based on the diagonal length of the clipping geometry's bounding box), and clips the resulting
    polygons to the union of geometries in the provided clipping GeoDataFrame. A small buffer is applied
    during the clipping process to prevent gaps between adjacent polygons. Additionally, a specified attribute
    from the input points is preserved in the output.

    Parameters
    ----------
    gdf_points : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries from which the Thiessen polygons will be generated.
    clipping_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the geometries that define the irregular boundary used for clipping the
        Thiessen polygons.
    preserve_attribute : str
        The name of the attribute column in `gdf_points` to retain in the output GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the clipped Thiessen polygons with the preserved attribute. The output
        GeoDataFrame maintains the same coordinate reference system (CRS) as the input `gdf_points`.

    Notes
    -----
    - The function first validates and cleans the clipping geometries to ensure a proper boundary.
    - An adaptive radius is computed based on the bounding box diagonal of the clipping union to extend
    infinite Voronoi regions.
    - The external function `voronoi_finite_polygons_2d` is used to convert infinite regions into finite polygons.
    - A slight buffer is applied during clipping to prevent gaps between adjacent polygons, and any remaining
    geometry artifacts are cleaned from the final output.
    """

    # Validate and prepare geometries
    clipping_union = unary_union(clipping_gdf.geometry).buffer(0)
    if not clipping_union.is_valid:
        clipping_union = clipping_union.buffer(0)

    # Calculate adaptive radius based on bounding box diagonal
    bounds = clipping_union.bounds
    diag_length = math.hypot(bounds[2]-bounds[0], bounds[3]-bounds[1])
    radius = 2.5 * diag_length  # Conservative radius

    # Generate Voronoi diagram
    points = np.array([[pt.x, pt.y] for pt in gdf_points.geometry])
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=radius)

    # Create and clean polygons
    polys = []
    for region in regions:
        poly = Polygon(vertices[region]).buffer(0)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polys.append(poly)

    # Create GeoDataFrame and clip
    thiessen_gdf = gpd.GeoDataFrame(
        gdf_points[[preserve_attribute]],
        geometry=polys,
        crs=gdf_points.crs
    )

    # Use overlay with small buffer to prevent gaps
    clip_mask = gpd.GeoDataFrame(geometry=[clipping_union.buffer(1e-6)], crs=thiessen_gdf.crs)
    thiessen_clipped = gpd.overlay(
        thiessen_gdf,
        clip_mask,
        how='intersection',
        keep_geom_type=False
    ).explode(index_parts=False).reset_index(drop=True)

    # Clean any remaining artifacts
    thiessen_clipped['geometry'] = thiessen_clipped.geometry.buffer(0)
    thiessen_clipped = thiessen_clipped[~thiessen_clipped.geometry.is_empty]

    #Merge polygons with same ID
    thiessen_clipped = thiessen_clipped.dissolve(by= preserve_attribute, as_index=False)
    
    return thiessen_clipped