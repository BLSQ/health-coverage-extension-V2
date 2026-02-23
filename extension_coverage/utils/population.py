from collections.abc import Sequence
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.merge
from rasterio.features import rasterize
from rasterstats import zonal_stats
from shapely.geometry import Polygon


def count_population_total(boundaries: gpd.GeoDataFrame, population: Path) -> pd.Series:
    """Count total population per area.

    Parameters
    ----------
    boundaries : geodataframe
        Input districts/areas geometries.
    population : Path
        Path to population raster (population per pixel).

    Returns
    -------
    series
        Population count per district/area.
    """
    with rasterio.open(str(population)) as src:
        stats = zonal_stats(
            raster=src.read(1),
            vectors=boundaries.geometry,
            stats=["sum"],
            affine=src.transform,
            nodata=src.nodata,
            all_touched=True,
        )

        pop_count = [stat["sum"] for stat in stats]

        boundaries["population_total"] = pop_count

        return boundaries


def calculate_population_covered(
    boundaries: gpd.GeoDataFrame,
    column_population_count: str,
    csi: gpd.GeoDataFrame,
    population: Path,
    distances: list[int] | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Compute population covered for each area.

    Parameters
    ----------
    boundaries : geodataframe
        Input areas.
    column_population_count: str
        Column name in boundaries with total population count for each area.
    csi : geodataframe
        Health centers.
    population : Path
        Path to population raster.
    distances : list of int
        Service distances in meters.
    output_dir : Path
        Path to the produced files.

    Returns
    -------
    dataframe
        Population covered count and ratio as a copy of the boundaries dataframe with added columns.
    """
    if distances is None:
        distances = [5000, 10000, 15000]

    with rasterio.open(str(population)) as src:
        pop = src.read(1)
        coverage = boundaries.copy()

        for distance in distances:
            data = []
            covered = csi.buffer(distance).union_all()

            for _, row in coverage.iterrows():
                covered_area = covered.intersection(row.geometry)

                if covered_area.area == 0 or not covered_area.is_valid:
                    data.append(0)
                    continue

                area_covered = rasterize(
                    [covered_area.__geo_interface__],
                    out_shape=pop.shape,
                    fill=0,
                    default_value=1,
                    dtype="uint8",
                    transform=src.transform,
                    all_touched=True,
                )

                pop_covered = pop[(area_covered == 1) & (pop != src.nodata)].sum()
                data.append(pop_covered)

            column = f"population_covered_{int(distance / 1000)}km"
            coverage[column] = data
            pop_covered_ratio = coverage[column] / boundaries[column_population_count]
            coverage[f"{column}_ratio"] = pop_covered_ratio

    # Save
    boundaries.to_file(output_dir / "population_coverage.gpkg", driver="GPKG", index=False)
    pd.DataFrame(boundaries.drop(columns=["geometry"])).to_csv(
        path_or_buf=output_dir / "population_coverage.csv", index=False
    )

    return coverage


# For generate_population_served
def get_kernel(population_raster: Path, buffer_size: int) -> np.ndarray:
    """Get an array kernel corresponding to a buffer of a given size.

    Generates a flat, disk-shaped footprint as 2d array which corresponds to a buffer area around each pixel.

    Parameters
    ----------
    population_raster : Path
        Path to population GeoTIFF.
    buffer_size : int
        Buffer size in meters.

    Returns
    -------
    kernel : ndarray
        Buffer kernel as a numpy array.
    """

    def _geom_from_bounds(bounds: Sequence[float]) -> Polygon:
        xmin, ymin, xmax, ymax = bounds
        return Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])

    with rasterio.open(str(population_raster)) as src:
        if src.crs.is_geographic:
            raise ValueError("Population raster must be in a projected CRS (meters).")

        extent = _geom_from_bounds(src.bounds)
        geom = extent.centroid.buffer(buffer_size)

        # Rasterize the buffer and use it as a 2d array kernel
        crop, crop_transform = rasterio.mask.mask(src, shapes=[geom], crop=True, indexes=1)
        kernel = rasterio.mask.geometry_mask([geom], out_shape=crop.shape, transform=crop_transform, invert=True)

    return kernel.astype("uint8")


def compute_population_served(population_raster: Path, area_served: int, geom: Polygon) -> np.ndarray:
    """Create a raster with population served per pixel.

    In the population served raster, each pixel is assigned the value corresponding to the total population count in
    a radius of `area_served` meters.

    More specifically, each pixel is assigned the value corresponding to the sum of all its neighboring pixels
    -- constrained by a disk-shaped footprint defined by the buffer area.

    Parameters
    ----------
    population_raster : Path
        Path to population GeoTIFF.
    area_served : int
        Area served radius in meters.
    geom : shapely geometry
        Area of interest. Pixel outside will be masked.

    Returns
    -------
    ndarray
        Population served per pixel.
    """
    kernel = get_kernel(population_raster, buffer_size=area_served)

    with rasterio.open(str(population_raster)) as src:
        pop = src.read(1)
        pop[pop < 0] = 0
        pop[pop == src.nodata] = 0
        area = rasterio.mask.geometry_mask(
            geometries=[geom.__geo_interface__],
            out_shape=pop.shape,
            transform=src.transform,
            all_touched=True,
            invert=True,
        )
        pop[~area] = 0

    pop_sum = cv2.filter2D(src=pop, ddepth=-1, kernel=kernel).astype("int32")
    pop_sum[~area] = -1
    return pop_sum


def generate_population_served(
    boundaries: gpd.GeoDataFrame, population_dir: Path, dst_file: Path, area_served: int
) -> Path:
    """Compute population served per pixel.

    In the population served raster, each pixel is assigned the value corresponding to the total population count in
    a radius of `area_served` meters.

    More specifically, each pixel is assigned the value corresponding to the sum of all its neighboring pixels --
    constrained by a disk-shaped footprint defined by the buffer area.

    Parameters
    ----------
    boundaries : geodataframe
        Geodataframe with districts (EPSG:32632).
    population_dir : Path
        Path to directory with splitted population rasters.
    dst_file : Path
        Path to output file.
    area_served : int
        Area served radius in meters.

    Returns
    -------
    Path
        Path to output file.

    Notes
    -----
    The functions uses GDAL, more specifically gdal_merge.py: <https://gdal.org/programs/gdal_merge.html>
    """
    tif_files = list(population_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {population_dir}")

    # Compute population served for each population raster tile, i.e. once per area
    for index, boundary in boundaries.iterrows():
        fp = population_dir / f"{index}_served.tif"
        population_raster = population_dir / f"{index}.tif"

        with rasterio.open(str(population_raster)) as src:
            dst_profile = src.profile.copy()

        pop_sum = compute_population_served(population_raster, area_served, boundary.geometry)

        dst_profile["dtype"] = "int32"
        dst_profile["nodata"] = -1
        dst_profile["BIGTIFF"] = "YES"

        with rasterio.open(str(fp), "w", **dst_profile) as dst:
            dst.write(pop_sum, 1)

    tiles = list(population_dir.glob("*_served.tif"))
    tiles_str = [str(tile) for tile in tiles]

    with rasterio.open(tiles_str[0]) as src:
        meta = src.meta.copy()

    data, dst_transform = rasterio.merge.merge(tiles_str)
    meta.update(
        {
            "driver": "GTiff",
            "count": data.shape[0],
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": dst_transform,
            "BIGTIFF": "YES",
        }
    )

    with rasterio.open(dst_file, "w", **meta) as dst:
        dst.write(data)

    return dst_file


def generate_priority_areas(
    population_served: Path,
    csi: gpd.GeoDataFrame,
    dst_file: Path,
    min_dist_from_csi: int = 15000,
) -> Path:
    """Compute a raster showing priority areas.

    Priority areas are locations at more than <buffer_size> of an existing CSI and with more than <population_threshold>
    people in a buffer of 5km (as defined in the population_served raster).

    Parameters
    ----------
    population_served : Path
        Path to population served raster.
    csi : geodataframe
        Geodataframe with CSI geometries.
    dst_file : Path
        Path to output file.
    min_dist_from_csi : int
        Buffer radius in meters. Minimum distance from existing CSI for priority areas.

    Returns
    -------
    Path
        Output file.
    """
    with rasterio.open(str(population_served)) as src:
        priority_areas = src.read(1)
        dst_profile = src.profile.copy()
        transform = src.transform
        width, height = src.width, src.height

    served_by_csi = rasterio.features.geometry_mask(
        geometries=[geom.__geo_interface__ for geom in csi.buffer(min_dist_from_csi)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )

    priority_areas[served_by_csi == 1] = -1
    with rasterio.open(str(dst_file), "w", **dst_profile) as dst:
        dst.write(priority_areas, 1)

    return dst_file


def population_served_per_fosa(
    fosa: gpd.GeoDataFrame, population_served_raster: Path, output_dir: Path, file_name: str
) -> pd.Series:
    """Get population served for each FOSA per area.

    Parameters
    ----------
    fosa : geodataframe
        Geodataframe with FOSAs.
    population_served_raster : Path
        Path to population served raster.
    output_dir: Path
        Output directory path.
    file_name: str
        Name of the file (either csi_..., or cs_...)

    Returns
    -------
    serie
        Population served for each FOSA.
    """
    fosa_pop_served = pd.Series(index=fosa.index, dtype="int32")

    with rasterio.open(str(population_served_raster)) as src:
        src_transform = src.transform
        pop_served = src.read(1)
        src_crs = src.crs

    if fosa.crs != src_crs:
        fosa_ = fosa.to_crs(src_crs)
    else:
        fosa_ = fosa.copy()

    for index, fs in fosa_.iterrows():
        if fs.geometry:
            row, col = rasterio.transform.rowcol(src_transform, fs.geometry.x, fs.geometry.y)

            try:
                fosa_pop_served.loc[index] = int(pop_served[row, col])
            except IndexError:
                fosa_pop_served.loc[index] = None

        else:
            fosa_pop_served.loc[index] = None

    fosa.to_file(output_dir / f"{file_name}.gpkg", driver="GPKG")
    pd.DataFrame(fosa.drop(columns=["geometry"])).to_csv(path_or_buf=output_dir / f"{file_name}.csv", index=False)

    return fosa_pop_served
