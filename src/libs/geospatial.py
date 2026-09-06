import ast
import os
import json
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import shapes, geometry_mask, geometry_window
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from shapely.geometry import Polygon, Point, mapping, box, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from pyproj import Transformer

from src.utils import context


config = context.config
logger = context.logger


def clip_raster_aoi_with_shapefile(raster_path, shapefile_path, output_path):
    """
    Function to clip raster with shapefile.
    :param raster_path:
    :param shapefile_path:
    :param output_path:
    :return:
    """

    shapefile = gpd.read_file(shapefile_path)

    # Open raster
    with rasterio.open(raster_path) as src:
        # Reproject shapefile to match raster CRS if needed
        if shapefile.crs != src.crs:
            shapefile = shapefile.to_crs(src.crs)

        # Get geometry in raster's CRS
        shapes = [mapping(geom) for geom in shapefile.geometry]

        # Mask the raster using the shapefile geometry
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs
        })

        # Write clipped raster to file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    logger.info(f"Clipped raster saved to: {output_path}")


def clip_raster_aoi_with_geojson(raster_path, geojson_path, output_path, block_rows=2048):
    """
    Clip raster using a GeoJSON file, streamed block-by-block.

    Equivalent to rasterio.mask(..., crop=True) but reads and writes the clipped
    extent in row-strip windows, so peak memory stays bounded (block_rows x width)
    instead of materialising the whole clipped array -- which OOMs on multi-GB
    drone bands. Pixels outside the AoI are set to the raster's nodata (0 when
    undefined), and the output is tiled/compressed/BIGTIFF-safe so a multi-GB
    clip does not overflow the classic-TIFF limit.

    :param raster_path: Path to input raster
    :param geojson_path: Path to GeoJSON file
    :param output_path: Path to output clipped raster
    :param block_rows: Rows read/written per streamed window
    """

    gdf = gpd.read_file(geojson_path)

    with rasterio.open(raster_path) as src:
        # Reproject GeoJSON to match raster CRS if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Convert geometries to GeoJSON-like mappings
        geoms = [mapping(geom) for geom in gdf.geometry if geom is not None]

        # Pixel window covering the AoI (the crop=True extent), snapped to whole
        # pixels and clamped to the raster.
        crop_window = geometry_window(src, geoms).round_offsets().round_lengths()
        crop_transform = src.window_transform(crop_window)
        crop_height = int(crop_window.height)
        crop_width = int(crop_window.width)

        fill_value = src.nodata if src.nodata is not None else 0

        out_profile = src.profile.copy()
        out_profile.update(
            driver="GTiff",
            height=crop_height,
            width=crop_width,
            transform=crop_transform,
            crs=src.crs,
            nodata=fill_value,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            bigtiff="IF_SAFER",
        )

        with rasterio.open(output_path, "w", **out_profile) as dest:
            for row_off in range(0, crop_height, block_rows):
                n_rows = min(block_rows, crop_height - row_off)

                # Window into the SOURCE (offset by the crop origin)
                src_window = Window(crop_window.col_off, crop_window.row_off + row_off,
                                    crop_width, n_rows)
                block = src.read(window=src_window)  # (bands, n_rows, crop_width)

                # Burn the AoI geometry for this block; True == outside the AoI
                block_transform = src.window_transform(src_window)
                outside = geometry_mask(geoms, out_shape=(n_rows, crop_width),
                                        transform=block_transform, invert=False)
                block[:, outside] = fill_value

                dest.write(block, window=Window(0, row_off, crop_width, n_rows))

    logger.info(f"Clipped raster saved to: {output_path}")


def geotiff_bounds_to_wgs84_bounds(geotiff_path):
    """
    Function to convert a geotiff's bounding box to WGS84 bounding box'
    :param geotiff_path:
    :return:
    """
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        crs = src.crs
        bounds_wgs84 = transform_bounds(crs, "EPSG:4326", *bounds)

    return bounds_wgs84


def create_h3_index_from_bounds(bounds_wgs84, resolution):
    """
    Function to create an h3 index from a bounding box
    :param bounds_wgs84:
    :param resolution:
    :return:
    """
    # Create a polygon box from bounds (lon, lat order for GeoJSON)
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [bounds_wgs84[0], bounds_wgs84[1]],  # lower-left
            [bounds_wgs84[0], bounds_wgs84[3]],  # upper-left
            [bounds_wgs84[2], bounds_wgs84[3]],  # upper-right
            [bounds_wgs84[2], bounds_wgs84[1]],  # lower-right
            [bounds_wgs84[0], bounds_wgs84[1]]   # close ring
        ]]
    }

    # Generate hexagons covering the polygon
    polygon = h3.geo_to_h3shape(polygon)
    h3_cells = h3.polygon_to_cells(polygon, res=resolution)

    return h3_cells


def h3_to_geodataframe(h3_cells):
    """
    Function to convert a h3 cell list into a geodataframe
    :param h3_cells:
    :return:
    """
    polygons = []
    indexes = []
    for h in h3_cells:
        boundary_latlon = h3.cell_to_boundary(h)
        boundary_lonlat = [(lon, lat) for lat, lon in boundary_latlon]
        poly = Polygon(boundary_lonlat)
        polygons.append(poly)
        indexes.append(h)
    result = gpd.GeoDataFrame({'h3_index': indexes, 'geometry': polygons}, crs="EPSG:4326")

    return result


def clip_raster_with_geojson(raster_path, geojson_path, output_dir):
    """
    Function to clip a raster with a geojson mask
    :param raster_path:
    :param geojson_path:
    :param output_dir:
    :return:
    """

    # Load GeoJSON polygons
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs("EPSG:4326")  # Ensure WGS84

    # Open the source raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

        # Reproject GeoJSON to match raster CRS
        gdf = gdf.to_crs(raster_crs)

        os.makedirs(output_dir, exist_ok=True)

        for i, row in gdf.iterrows():
            geometry = row.geometry
            try:
                out_image, out_transform = mask(src, [mapping(geometry)], crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": raster_crs
                })

                out_path = os.path.join(output_dir, f"hextile_{i}.tif")
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                logger.info(f"Saved clipped polygon/raster to: {out_path}")
            except Exception as e:
                logger.error(f"Skipping polygon/raster {i} due to error: {e}")


def raster_statistics_with_geojson(raster_path, geojson_path, output_dir,  statistic='mean', channel=None):
    """
    Function to calculate raster statistics with a geojson mask
    :param raster_path:
    :param geojson_path:
    :param output_dir:
    :return:
    """

    # Load GeoJSON polygons
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs("EPSG:4326")  # Ensure WGS84

    # Open the source raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

        # Reproject GeoJSON to match raster CRS
        gdf = gdf.to_crs(raster_crs)

        os.makedirs(output_dir, exist_ok=True)

        for i, row in gdf.iterrows():
            geometry = row.geometry
            try:
                out_image, out_transform = mask(src, [mapping(geometry)], crop=True)

                if channel is not None:
                    out_image = out_image[channel, :, :]

                if statistic == 'mean':
                    gdf_statistic = np.mean(out_image)
                else:
                    gdf_statistic = statistic(out_image)

                gdf.loc[i, f'statistic_{statistic}'] = gdf_statistic

                if gdf_statistic <= 0.0 :
                    gdf.loc[i, f'open_canopy'] = 0
                elif (gdf_statistic >= 0.0) and (gdf_statistic < 0.1):
                    gdf.loc[i, f'open_canopy'] = 1
                elif gdf_statistic >= 0.1:
                    gdf.loc[i, f'open_canopy'] = 0
                else:
                    pass

                logger.info(f"Computing {statistic} stats for raster tile: {i}")
            except Exception as e:
                logger.error(f"Skipping polygon/raster {i} due to error: {e}")

    gdf.to_file(output_dir + '/h3_stats.geojson', driver="GeoJSON")
    logger.info(f"Written {statistic} stats for all tiles to: {output_dir}")

    return gdf


def filter_polygons_by_raster_data(raster_path, geojson_path, output_geojson_path):
    """
    Function to filter a polygon by raster data
    :param raster_path:
    :param geojson_path:
    :param output_geojson_path:
    :return:
    """

    # Load GeoJSON polygons
    gdf = gpd.read_file(geojson_path)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Reproject GeoJSON to match raster CRS
        gdf = gdf.to_crs(raster_crs)

        kept_polygons = []

        for idx, row in gdf.iterrows():
            geometry = row.geometry
            try:
                out_image, _ = mask(src, [mapping(geometry)], crop=True)
                if nodata is not None:
                    if (out_image != nodata).any():
                        kept_polygons.append(row)
                else:
                    if out_image.any():
                        kept_polygons.append(row)
            except Exception as e:
                logger.error(f"Skipping feature {idx} due to error: {e}")
                continue

    # Create new GeoDataFrame with only kept polygons
    if kept_polygons:
        filtered_gdf = gpd.GeoDataFrame(kept_polygons, crs=raster_crs)
        filtered_gdf.to_crs("EPSG:4326").to_file(output_geojson_path, driver="GeoJSON")
        logger.info(f"Saved {len(filtered_gdf)} polygons to '{output_geojson_path}'")
    else:
        logger.error("No polygons intersect valid raster data.")


def filter_gpd_by_raster_data(raster_path, geojson_data, output_geojson_path):
    """
    Function to filter a geopandas dataframe by raster data
    :param raster_path:
    :param geojson_data:
    :param output_geojson_path:
    :return:
    """
    # Load GeoJSON polygons
    gdf = geojson_data

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Reproject GeoJSON to match raster CRS
        gdf = gdf.to_crs(raster_crs)

        kept_polygons = []

        for idx, row in gdf.iterrows():
            geometry = row.geometry
            try:
                out_image, _ = mask(src, [mapping(geometry)], crop=True)
                if nodata is not None:
                    if (out_image != nodata).any():
                        kept_polygons.append(row)
                else:
                    if out_image.any():
                        kept_polygons.append(row)
            except Exception as e:
                logger.error(f"Skipping feature {idx} due to error: {e}")
                continue

    # Create new GeoDataFrame with only kept polygons
    if kept_polygons:
        filtered_gdf = gpd.GeoDataFrame(kept_polygons, crs=raster_crs)
        filtered_gdf.to_crs("EPSG:4326").to_file(output_geojson_path, driver="GeoJSON")
        logger.info(f"Saved {len(filtered_gdf)} polygons to '{output_geojson_path}'")
    else:
        logger.error("No polygons intersect valid raster data.")


def transform_survlatlon_to_geojson(latlon_points, tiff_src, output_geojson_path):
    """
    Transform latitude and longitude points to geojson dataframe
    :param latlon_points:
    :param tiff_src:
    :param output_geojson_path:
    :return:
    """
    # Set up transformer from WGS84 to the TIFF CRS
    transformer = Transformer.from_crs("EPSG:4326", tiff_src.crs, always_xy=True)

    # Create bounding box polygon in TIFF CRS
    tiff_bbox = box(*tiff_src.bounds)

    filtered_geometries = []
    for lat, lon in latlon_points:
        y, x = transformer.transform(lon, lat)
        point = Point(x, y)
        if tiff_bbox.contains(point):
            filtered_geometries.append(point)

    # Create GeoDataFrame with filtered points
    gdf = gpd.GeoDataFrame(geometry=filtered_geometries, crs=tiff_src.crs)
    gdf.to_crs("EPSG:4326").to_file(output_geojson_path, driver="GeoJSON")
    logger.info(f"Saved {len(gdf)} points to {output_geojson_path}")

    return gdf


def transform_survraster_to_geojson(tiff_src, geojson_output, tile_id, tile_value, attribute_name, attribute_value):
    """

    :param tiff_src:
    :param geojson_output:
    :param tile_id:
    :param tile_value:
    :param attribute_name:
    :param attribute_value:
    :return:
    """

    if isinstance(tiff_src, str):
        tiff_src = rasterio.open(tiff_src)

    mask = tiff_src.dataset_mask()  # Valid data mask
    # Extract shapes of valid data areas
    geoms = [shape(geom) for geom, val in shapes(mask, mask > 0, transform=tiff_src.transform)]
    # Merge all shapes into one boundary (union)
    boundary = gpd.GeoSeries(geoms).union_all()
    # Create GeoDataFrame with attribute
    gdf = gpd.GeoDataFrame([{
        tile_id: tile_value,
        attribute_name: attribute_value,
        'geometry': boundary}], crs=tiff_src.crs)

    # Save to GeoJSON in WGS84
    gdf.to_crs("EPSG:4326").to_file(geojson_output, driver="GeoJSON")
    logger.info(f"Boundary saved to: {geojson_output}")

    return gdf


def combine_geodataframes(gdf_list):
    """

    :param gdf_list:
    :return:
    """
    # Ensure all GeoDataFrames share the same CRS
    common_crs = gdf_list[0].crs
    for i, gdf in enumerate(gdf_list):
        if gdf.crs != common_crs:
            gdf_list[i] = gdf.to_crs(common_crs)

    # Combine into a single GeoDataFrame
    combined_gdf = pd.concat(gdf_list, ignore_index=True)

    return gpd.GeoDataFrame(combined_gdf, crs=common_crs)


def convert_boundary_to_geojson_file(df: pd.DataFrame,
                                     key_col:str='planting_session_id',
                                     polygon_col:str='boundary',
                                     output_dir: str='./',
                                     verbose=False):
    """
    Convert boundary(ies) of planting sessions from field_updates to GeoJSON file(s)
    :param df:
    :param key_col: main column to use for the session ids
    :param output_path: directory to save the GeoJSON file(s)
    :param verbose: log progress to stdout
    :return:
    """

    # os.makedirs(output_path, exist_ok=True)

    for _, row in df.iterrows():
        session_id = row[key_col]

        try:
            geofence_dict = json.loads(row[polygon_col].replace("'", '"'))
        except Exception as e:
            print(f"Error parsing geofence for session {session_id}: {e}")
            continue

        # Extract the actual geometry (assuming structure like {"geofence": {...}})
        geometry = geofence_dict.get("geofence", {})

        # TODO: can insert a 'season' property here as well - TBD
        geojson_feature = {
            "type": "Feature",
            "properties": {
                "planting_session_id": session_id,
                "date_planted": row.get("date_planted"),
                "started_planting_at": row.get("started_planting_at"),
                "finished_planting_at": row.get("finished_planting_at"),
                "planting_site": row.get("subsite_prefix_name"),
                "region": row.get("planting_site_prefix_name"),
                "amount_planted": row.get("amount_planted"),
            },
            "geometry": geometry
        }

        geojson = {
            "type": "FeatureCollection",
            "features": [geojson_feature]
        }

        if verbose:
            logger.info(f"Processed session ... {session_id}")

        output_path = os.path.join(output_dir, f"{session_id}.geojson")

        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

    return None


def _parse(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return ast.literal_eval(val)
    return val


def _extract_geom(val):
    """Dig out a GeoJSON geometry dict from common wrappers."""
    val = _parse(val)
    if not isinstance(val, dict):
        return None
    # unwrap {"geofence": ...}, {"geometry": ...}, Feature, FeatureCollection
    if "geofence" in val:
        val = _parse(val["geofence"])
    if isinstance(val, dict) and val.get("type") == "Feature":
        val = val.get("geometry")
    if isinstance(val, dict) and val.get("type") == "FeatureCollection":
        feats = val.get("features") or []
        return [shape(f["geometry"]) for f in feats if f.get("geometry")]
    return shape(val) if isinstance(val, dict) and "type" in val else None


# def filter_intersecting(df, master_geojson_path, geom_col="geofence_data", debug=False):
#     """
#     Filters rows in a DataFrame based on whether their geometries intersect with a
#     master geometry defined in a GeoJSON file.
#
#     :param df: The input GeoDataFrame containing the data to be filtered.
#     :type df: geopandas.GeoDataFrame
#     :param master_geojson_path: The file path to the GeoJSON file containing the
#         master geometries to use for intersection filtering.
#     :type master_geojson_path: str
#     :param geom_col: The column in the input DataFrame containing geometric data.
#         Defaults to "geofence_data".
#     :type geom_col: str, optional
#     :param debug: A flag to enable detailed debug output that logs intermediate
#         processing details. Defaults to False.
#     :type debug: bool, optional
#     :return: A filtered copy of the input DataFrame containing only rows where
#         the geometries intersect with the master geometry.
#     :rtype: geopandas.GeoDataFrame
#     """
#
#     master = gpd.read_file(master_geojson_path)
#     master_geom = make_valid(unary_union(master.geometry))
#
#     # master_union = make_valid(unary_union(master.geometry.buffer(0)))
#     # master_geom = master_union.envelope
#
#     def to_shape(val):
#         try:
#             g = _extract_geom(val)
#         except Exception as e:
#             if debug: print("parse fail:", e)
#             return None
#         if g is None:
#             return None
#         if isinstance(g, list):
#             g = unary_union(g)
#         if not g.is_valid:
#             g = make_valid(g)
#         return g
#
#     geoms = df[geom_col].map(to_shape)
#
#     if debug:
#         print("rows:", len(df),
#               "parsed:", geoms.notna().sum(),
#               "master_bounds:", master_geom.bounds)
#         sample = geoms.dropna().head(3)
#         for i, g in sample.items():
#             print(i, g.geom_type, g.bounds)
#
#     mask = geoms.map(lambda g: g is not None and g.intersects(master_geom))
#     return df[mask].copy()


def filter_intersecting(df, master_geojson_path, geom_col="geofence_data",
                        min_overlap=0.0, overlap_of="row", debug=False):
    """
    Filter rows whose geometry overlaps the master by at least `min_overlap`.

    :param min_overlap: fraction in [0, 1]. 0.0 = any intersection, 1.0 = fully
        contained.
    :param overlap_of: "row"    → intersection_area / row_area (default),
                       "master" → intersection_area / master_area,
                       "iou"    → intersection / union.
    """
    assert 0.0 <= min_overlap <= 1.0
    assert overlap_of in {"row", "master", "iou"}

    master = gpd.read_file(master_geojson_path)
    master_geom = make_valid(unary_union(master.geometry))
    master_area = master_geom.area

    def to_shape(val):
        try:
            g = _extract_geom(val)
        except Exception as e:
            if debug: print("parse fail:", e)
            return None
        if g is None:
            return None
        if isinstance(g, list):
            g = unary_union(g)
        if not g.is_valid:
            g = make_valid(g)
        return g

    def overlap_frac(g):
        if g is None or not g.intersects(master_geom):
            return 0.0
        inter = g.intersection(master_geom).area
        if overlap_of == "row":
            denom = g.area
        elif overlap_of == "master":
            denom = master_area
        else:  # iou
            denom = g.union(master_geom).area
        return inter / denom if denom > 0 else 0.0

    geoms = df[geom_col].map(to_shape)
    fracs = geoms.map(overlap_frac)

    if debug:
        print("rows:", len(df),
              "parsed:", geoms.notna().sum(),
              "master_bounds:", master_geom.bounds,
              "master_area:", master_area)
        print("overlap distribution:", fracs.describe())

    # use >= for 0.0 (keep any intersection) so passing 0.0 matches old behavior
    mask = fracs >= min_overlap if min_overlap > 0 else fracs > 0
    return df[mask].copy()


def georeference_tiff(src_path, dst_path, bounds, crs="EPSG:4326"):
    """
    bounds = (west, south, east, north) in the target CRS.
    For WGS84 lon/lat: (min_lon, min_lat, max_lon, max_lat).
    """
    with rasterio.open(src_path) as src:
        data = src.read()
        profile = src.profile.copy()

    transform = from_bounds(*bounds, width=profile["width"], height=profile["height"])
    profile.update(crs=crs, transform=transform, driver="GTiff")

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)


def count_points_per_polygon(points_geojson, polygons_dir, recursive=False):
    """
    Count how many points from `points_geojson` fall inside each polygon file
    in `polygons_dir`.

    Returns a DataFrame with columns: file, polygon_index, point_count.
    """
    points = gpd.read_file(points_geojson)
    if points.empty:
        return pd.DataFrame(columns=["file", "polygon_index", "point_count"])

    pattern = "**/*.geojson" if recursive else "*.geojson"
    rows = []

    for path in sorted(Path(polygons_dir).glob(pattern)):
        polys = gpd.read_file(path)
        if polys.empty:
            continue

        # align CRS
        if polys.crs and points.crs and polys.crs != points.crs:
            pts = points.to_crs(polys.crs)
        else:
            pts = points

        # spatial join: each point gets the index of every polygon it lies in
        joined = gpd.sjoin(pts, polys[["geometry"]], how="inner", predicate="intersects")
        counts = joined.groupby("index_right").size()

        # include polygons with zero hits
        counts = counts.reindex(polys.index, fill_value=0)

        for poly_idx, n in counts.items():
            rows.append({
                "file": path.name,
                "polygon_index": int(poly_idx),
                "point_count": int(n),
            })

    return pd.DataFrame(rows)


def planting_per_polygon(planting_geojson_dir):
    """
    Returns a DataFrame with columns: planting_session_id, amount_planted.
    """

    planting_geojson_files = os.listdir(planting_geojson_dir)

    rows = []
    for i_file, _ in enumerate(planting_geojson_files):

        df_gpd = gpd.read_file(planting_geojson_dir + planting_geojson_files[i_file])

        try:
            rows.append({
                "planting_session_id": df_gpd['planting_session_id'].values[0],
                "amount_planted": df_gpd['amount_planted'].values[0],
                "finished_planting_at": df_gpd['finished_planting_at'].values[0],
            })
        except Exception as e:
            print(f"Error processing row {i_file}: {e}")
            continue

    return pd.DataFrame(rows)


def geojson_area_m2(filepath):
    """
    Get Area of a GeoJSON polygon file
    filepath: path to GeoJSON file
    returns: total area (m²) of all polygons in the file
    """
    gdf = gpd.read_file(filepath)

    # Ensure CRS is set (assume WGS84 if missing)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    # Project to appropriate UTM zone for accuracy
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    return gdf_proj.geometry.area.sum()


def raster_to_geojson(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    target_crs: str = "EPSG:4326",
) -> dict:
    """Return a GeoJSON Feature whose polygon is the raster's bounding box.

    :param raster_path: input raster (any GDAL-readable format)
    :param output_path: optional path to write the GeoJSON
    :param target_crs: GeoJSON spec is EPSG:4326; override only if you know why
    :return: GeoJSON FeatureCollection with one rectangular polygon
    """
    with rasterio.open(raster_path) as src:
        # densify_pts curves the edges so the reprojected polygon hugs the
        # true extent rather than degenerating to four straight lines.
        left, bottom, right, top = transform_bounds(
            src.crs, target_crs, *src.bounds, densify_pts=21
        )

    poly = box(left, bottom, right, top)

    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(poly)}],
    }

    if output_path is not None:
        Path(output_path).write_text(json.dumps(fc))

    return fc


def merge_tiffs(input_paths, output_path, nodata=None, method="first"):
    """Merge a list of raster TIFFs into a single GeoTIFF (mosaic).

    Args:
        input_paths: Iterable of paths to the input .tif files.
        output_path: Path to write the merged GeoTIFF.
        nodata: Value to treat as "no data" in inputs and to write in the
            output. If None, the nodata value of the first input is used.
        method: How to resolve overlapping pixels. One of rasterio's merge
            methods: "first" (default), "last", "min", "max".

    Returns:
        The output path as a Path.

    Raises:
        ValueError: If input_paths is empty.
    """
    input_paths = [str(p) for p in input_paths]
    if not input_paths:
        raise ValueError("input_paths must contain at least one file")

    # Open all sources; rasterio.merge reads them together into one array.
    sources = [rasterio.open(p) for p in input_paths]
    try:
        mosaic, out_transform = merge(sources, nodata=nodata, method=method)

        # Base the output profile on the first source, then override the
        # fields that change after merging. BIGTIFF="IF_SAFER" lets GDAL switch
        # to the BigTIFF format whenever the mosaic would exceed the classic
        # 4GB TIFF limit, while keeping smaller layers as standard GeoTIFFs.
        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=out_transform,
            compress="lzw",
            BIGTIFF="IF_SAFER",
        )
        if nodata is not None:
            profile["nodata"] = nodata

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in sources:
            src.close()

    return output_path


def merge_geojsons(input_paths, output_path):
    """Merge (union) multiple GeoJSON files into a single GeoJSON.

    Args:
        input_paths: Iterable of paths to the input .geojson files.
        output_path: Path to write the merged GeoJSON.

    Returns:
        The output path as a Path.

    Raises:
        ValueError: If input_paths is empty.
    """
    input_paths = [str(p) for p in input_paths]
    if not input_paths:
        raise ValueError("input_paths must contain at least one file")

    # Collect every geometry, reprojecting to the first file's CRS so the union
    # is computed in a single coordinate system.
    geometries = []
    target_crs = None
    for i_path in input_paths:
        gdf = gpd.read_file(i_path)
        if target_crs is None:
            target_crs = gdf.crs
        elif gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        geometries.extend(geom for geom in gdf.geometry if geom is not None)

    merged_geometry = unary_union(geometries)
    merged = gpd.GeoDataFrame(geometry=[merged_geometry], crs=target_crs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_path, driver="GeoJSON")

    logger.info(f"Merged GeoJSON saved to: {output_path}")

    return output_path
