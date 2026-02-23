from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from openhexa.sdk import current_run


def merge_districts(df_shapes: gpd.GeoDataFrame, output_dir: Path) -> gpd.GeoDataFrame:
    """Merge all geometries of districts to get national boundaries.

    Parameter
    ---------
    df_shapes: gpd.GeoDataFrame
        GeoDatFrame with valid geometries
    output_dir: Path
        Ouput directory path were files will be stored.

    Returns
    -------
    regions: gpd.GeoDataFrame
        Geodataframe of regions
    """
    country = gpd.GeoDataFrame(geometry=[df_shapes.geometry.union_all()], crs=df_shapes.crs)
    regions = df_shapes.dissolve("level_2_name", as_index=False).drop(columns=["level_3_name", "level_3_id"], axis=1)

    country.to_file(output_dir / "country.gpkg", driver="GPKG")
    regions.to_file(output_dir / "regions.gpkg", driver="GPKG")

    return regions


def save_buffered_geom(output_dir: Path, health_facilities: gpd.GeoDataFrame, name: str, buffers: list):
    """Add buffer(s) around geometry.

    Parameters
    ----------
    output_dir: Path
        Path to output directory.
    health_facilities: gpd.GeoDataFrame
        Geodataframe with health facilities (CSI/CS)
    name: str
        Used in filename (either csi or cs)
    buffers: list
        List of buffers (in meters) to compute around geometries
    """
    # Note: merge all geometries because there would be gaps when filtering by district,
    # as some CSI are not located within the geom of their parent
    for dist in buffers:
        # CSI
        gpd.GeoDataFrame(geometry=health_facilities.buffer(dist)).dissolve().to_file(
            output_dir / f"csi_buffer_{dist // 1000}km.gpkg", driver="GPKG"
        )


def split_raster(raster: Path, boundaries: gpd.GeoDataFrame, level: str, output_dir: Path) -> None:
    """Split population raster per area.

    The function split the input population raster into multiple tiles (one per area). This is to avoid
    taking into account population from other area in the following computations.

    Parameters
    ----------
    raster : Path
        Path to population raster.
    boundaries : geodataframe
        Health boundaries.
    level: str
        Either region or district
    output_dir : Path
        Path to output directory.
    """
    current_run.log_info(f"Génère les tiles de population pour chaque {level}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raster.is_file():
        raise ValueError(f"Population raster not found at {raster.as_posix()}")

    with rasterio.open(str(raster)) as src:
        # Reproject districts geodataframe if needed
        boundaries_ = boundaries.copy()
        if boundaries_.crs != src.crs:
            boundaries_ = boundaries_.to_crs(src.crs)

        dst_profile = src.profile
        dst_profile["compress"] = "zstd"
        dst_profile["predictor"] = 3

        for index, boundary in boundaries_.iterrows():
            fp = output_dir / f"{index}.tif"

            # Read a window of the population raster based on the geometry.
            window = rasterio.mask.geometry_window(src, shapes=[boundary.geometry.__geo_interface__])
            transform = src.window_transform(window)
            population = src.read(1, window=window)

            # Make sure that pixels outside the area are assigned the nodata value.
            mask_ = rasterio.mask.geometry_mask(
                geometries=[boundary.geometry.__geo_interface__],
                out_shape=population.shape,
                transform=transform,
                all_touched=False,
                invert=True,
            )
            population[~mask_] = src.nodata

            # Write raster tile to disk with area index as name
            dst_profile["transform"] = transform
            dst_profile["width"] = population.shape[1]
            dst_profile["height"] = population.shape[0]
            dst_profile["dtype"] = "float32"
            with rasterio.open(str(fp), "w", **dst_profile) as dst:
                dst.write(population.astype(np.float32), 1)
