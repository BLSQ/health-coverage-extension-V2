import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.merge
from openhexa.sdk import current_run, pipeline, workspace
from rasterio.features import rasterize
from rasterstats import zonal_stats
from shapely.geometry import Polygon, shape
from health_coverage_map import DistrictHealthCoverageMap


@pipeline("extension_coverage")
def extension_coverage():
    """Compute health service coverage and identify potential extension areas.

    This pipeline performs a complete spatial analysis to assess health coverage
    based on population distribution and the location of health facilities.
    It includes the following steps:

    - Load health facilities (CS, CSI), administrative boundaries (districts),
      and population raster data.
    - Generate buffer areas around CS and CSI for predefined distance thresholds.
    - Compute total population per district.
    - Estimate population coverage per district for multiple service distances.
    - Split the population raster by district to generate analysis tiles.
    - Calculate the population currently served within a maximum service distance.
    - Identify priority areas for extension based on distance from existing CSI.
    - Compute population served by each CSI and CS.
    - Analyse potential extension areas and the extension potential of CS,
      based on minimum population thresholds and spatial constraints.
    - Parse and zip all files within their district folder
    - Upload zipped folder into s3 bucket
    """
    org_unit_dir = Path(workspace.files_path) / "organisation_units"

    # Load org units
    cs = gpd.read_file(org_unit_dir / "CS.gpkg")
    csi = gpd.read_file(org_unit_dir / "CSI.gpkg")
    districts = gpd.read_file(org_unit_dir / "shapes_level3.gpkg")

    # Population path
    population_path = Path(workspace.files_path) / "population/population.tif"

    # Dirs def
    results_dir = Path(workspace.files_path) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir = results_dir / "buffer_areas"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = results_dir / "intermediary"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    calculus_path = results_dir / "calculus"
    calculus_path.mkdir(parents=True, exist_ok=True)
    final_path = results_dir / "final_folder"
    final_path.mkdir(parents=True, exist_ok=True)

    # Get national boundaries from the merge of districts geometries
    # NB: Used during the atlas generation
    country = merge_districts(districts)
    country.to_file(buffer_dir / "country.gpkg", driver="GPKG")

    # Write buffer areas in output directory
    (results_dir / "buffer_areas").mkdir(parents=True, exist_ok=True)
    for dist in (5, 15):
        # CSI
        gpd.GeoDataFrame(geometry=csi.buffer(dist * 1000)).dissolve().to_file(
            buffer_dir / f"csi_buffer_{dist}km.gpkg", driver="GPKG"
        )

        # CS
        gpd.GeoDataFrame(geometry=cs.buffer(dist * 1000)).dissolve().to_file(
            buffer_dir / f"cs_buffer_{dist}km.gpkg", driver="GPKG"
        )

    # Calculs
    current_run.log_info("Calcule la population totale par district...")
    pop_total = count_population_total(boundaries=districts, population=population_path)
    districts["population_total"] = pop_total

    current_run.log_info("Calcule la couverture sanitaire pour chaque district...")
    districts = calculate_population_covered(
        boundaries=districts,
        column_population_count="population_total",
        csi=csi,
        population=population_path,
        distances=[5000, 10000, 15000],
    )

    districts.to_file(calculus_path / "population_coverage.gpkg", driver="GPKG", index=False)
    pd.DataFrame(districts.drop(columns=["geometry"])).to_csv(
        path_or_buf=calculus_path / "population_coverage.csv", index=False
    )

    current_run.log_info("Génère les tiles de population pour chaque district...")
    output_tiles = results_dir / "tiles"
    output_tiles.mkdir(parents=True, exist_ok=True)
    _ = split_population_raster(population_raster=population_path, districts=districts, output_dir=output_tiles)

    current_run.log_info("Calcule la population desservie...")
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    dst_file = intermediate_dir / "population_served.tif"
    max_distance_served = 5000
    # mx_distance_served is a param in the older version. TODO: we would like it to be either 5, 10 or 15km.
    served = generate_population_served(
        districts=districts, population_dir=output_tiles, dst_file=dst_file, area_served=max_distance_served
    )

    current_run.log_info("Génère les zones d'extension potentielles...")
    dst_file = intermediate_dir / "priority_areas.tif"
    min_distance_from_csi = 15000
    # min_distance_from_csi is a param in the older version. TODO: we would like it to be either 5, 10 or 15km.
    priority_areas = generate_priority_areas(
        population_served=served,
        csi=csi,
        raster_template=served,
        dst_file=dst_file,
        min_dist_from_csi=min_distance_from_csi,
    )

    current_run.log_info("Calcule la population desservie par chaque CSI...")
    column = f"population_{int(max_distance_served / 1000)}km"
    csi[column] = population_served_per_fosa(fosa=csi, population_served_raster=served)

    csi.to_file(calculus_path / "csi_population_served.gpkg", driver="GPKG")
    pd.DataFrame(csi.drop(columns=["geometry"])).to_csv(
        path_or_buf=calculus_path / "csi_population_served.csv", index=False
    )

    current_run.log_info("Calcule la population desservie par chaque CS...")
    cs[column] = population_served_per_fosa(fosa=cs, population_served_raster=served)
    cs.to_file(calculus_path / "cs_population_served.gpkg", driver="GPKG")
    pd.DataFrame(cs.drop(columns=["geometry"])).to_csv(
        path_or_buf=calculus_path / "cs_population_served.csv", index=False
    )

    current_run.log_info("Analyse les zones potentielles d'extension...")
    min_population = 5000
    # min_population is a param in the older version. TODO: we would like it to be either 5k, 6k, 7k...
    potential_areas = analyse_potential_areas(priority_areas=priority_areas, csi=csi, min_population=min_population)

    if not potential_areas.empty:
        potential_areas.to_file(calculus_path / "extension_areas.gpkg", driver="GPKG")

    current_run.log_info("Analyse le potentiel d'extension des CS...")
    potential_cs = analyse_cs(cs=cs, csi=csi, districts=districts, column=column)

    potential_cs.to_file(calculus_path / "cs_extension_potential.gpkg", driver="GPKG")
    pd.DataFrame(potential_cs.drop(columns=["geometry"])).to_csv(
        calculus_path / "cs_extension_potential.csv", index=False
    )

    current_run.log_info("Modélisation terminée ! Préparation du dossier contenant les fichiers de sortie... ")

    # Parse all files according to the district
    files_paths = get_files_paths(
        dirs_path=[buffer_dir, calculus_path], files_path=[results_dir / "healthcoverage_atlas.qgz"]
    )

    unzip_folders = final_path / "unzip"
    _ = split_files_by_district(
        file_paths=files_paths, districts=districts, district_col="level_3_name", output_dir=unzip_folders
    )

    # Generate pdf atlas
    current_run.log_info("Génère le PDF atlas pour chaque district...")

    # Parcours des dossiers de district
    for district_folder in unzip_folders.iterdir():
        if not district_folder.is_dir():
            continue

        district_name = district_folder.name
        current_run.log_info(f"Traitement du district : {district_name}")

        # Chemins des fichiers nécessaires
        population_coverage_gpkg = district_folder / "population_coverage.gpkg"
        cs_population_served_gpkg = district_folder / "cs_population_served.gpkg"
        csi_population_served_gpkg = district_folder / "csi_population_served.gpkg"
        cs_extension_potential_gpkg = district_folder / "cs_extension_potential.gpkg"
        buffer_dir = district_folder / "buffer_areas"
        country_gpkg = buffer_dir / "country.gpkg"

        # Instanciation de la classe
        district_map = DistrictHealthCoverageMap(output_dir=district_folder)

        # Génération du PDF
        pdf_path = district_map.generate(
            population_coverage=population_coverage_gpkg,
            csi_population_served=csi_population_served_gpkg,
            cs_population_served=cs_population_served_gpkg,
            cs_extension_potential=cs_extension_potential_gpkg,
            csi_buffer_5km=buffer_dir / "csi_buffer_5km.gpkg",
            csi_buffer_15km=buffer_dir / "csi_buffer_15km.gpkg",
            extension_areas=cs_extension_potential_gpkg,
            country=country_gpkg,
        )

        current_run.log_info(f"PDF généré pour {district_name} : {pdf_path.name}")


    # Zip and upload to s3 bucket to make docs accessible through the interface cartesanitaireniger.org
    # _ = zip_folder(dir_path=unzip_folders, output_dir=final_path / "zip")
    # _ = upload_to_s3(folder_path=final_path / "zip")


def merge_districts(df_shapes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge all geometries of districts to get national boundaries.

    Parameter
    ---------
    df_shapes: gpd.GeoDataFrame
        GeoDatFrame with valid geometries

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of one line with national geometry
    """
    geom = df_shapes.geometry.union_all()

    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32632")


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
        return pd.Series(data=pop_count, index=boundaries.index)


def calculate_population_covered(
    boundaries: gpd.GeoDataFrame,
    column_population_count: str,
    csi: gpd.GeoDataFrame,
    population: Path,
    distances: list[int] | None,
) -> pd.DataFrame:
    """Compute population covered for each district.

    Parameters
    ----------
    boundaries : geodataframe
        Input districts/areas.
    column_population_count: str
        Column name in boundaries with total population count for each area.
    csi : geodataframe
        Health centers.
    population : Path
        Path to population raster.
    distances : list of int
        Service distances in meters.

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
                covered_district = covered.intersection(row.geometry)

                if covered_district.area == 0 or not covered_district.is_valid:
                    data.append(0)
                    continue

                area_covered = rasterize(
                    [covered_district.__geo_interface__],
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

    return coverage


def split_population_raster(population_raster: Path, districts: gpd.GeoDataFrame, output_dir: Path) -> None:
    """Split population raster per district.

    The function split the input population raster into multiple tiles (one per district). This is to avoid
    taking into account population from other districts in the following computations.

    Parameters
    ----------
    population_raster : Path
        Path to population raster.
    districts : geodataframe
        Health districts.
    output_dir : Path
        Path to output directory.
    """
    if not population_raster.is_file():
        raise ValueError(f"Population raster not found at {population_raster.as_posix()}")

    with rasterio.open(str(population_raster)) as src:
        # Reproject districts geodataframe if needed
        districts_ = districts.copy()
        if districts_.crs != src.crs:
            districts_ = districts_.to_crs(src.crs)

        dst_profile = src.profile
        dst_profile["compress"] = "zstd"
        dst_profile["predictor"] = 3

        for index, district in districts_.iterrows():
            fp = output_dir / f"{index}.tif"

            # Read a window of the population raster based on the district geometry.
            window = rasterio.mask.geometry_window(src, shapes=[district.geometry.__geo_interface__])
            transform = src.window_transform(window)
            population = src.read(1, window=window)

            # Make sure that pixels outside the district are assigned the nodata value.
            mask_ = rasterio.mask.geometry_mask(
                geometries=[district.geometry.__geo_interface__],
                out_shape=population.shape,
                transform=transform,
                all_touched=False,
                invert=True,
            )
            population[~mask_] = src.nodata

            # Write raster tile to disk with district index as name
            dst_profile["transform"] = transform
            dst_profile["width"] = population.shape[1]
            dst_profile["height"] = population.shape[0]
            dst_profile["dtype"] = "float32"
            with rasterio.open(str(fp), "w", **dst_profile) as dst:
                dst.write(population.astype(np.float32), 1)


# For generate_population_served
def _geom_from_bounds(bounds: Sequence[float]) -> Polygon:
    """Convert a bounding box into a Shapely Polygon.

    Parameters
    ----------
    bounds : Sequence[float]
        A sequence of four floats (xmin, ymin, xmax, ymax) representing the bounding box coordinates.

    Returns
    -------
    Polygon
        A Shapely Polygon representing the bounding box.
    """
    xmin, ymin, xmax, ymax = bounds
    return Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])


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
        district = rasterio.mask.geometry_mask(
            geometries=[geom.__geo_interface__],
            out_shape=pop.shape,
            transform=src.transform,
            all_touched=True,
            invert=True,
        )
        pop[~district] = 0

    pop_sum = cv2.filter2D(src=pop, ddepth=-1, kernel=kernel).astype("int32")
    pop_sum[~district] = -1
    return pop_sum


def generate_population_served(
    districts: gpd.GeoDataFrame, population_dir: Path, dst_file: Path, area_served: int
) -> Path:
    """Compute population served per pixel.

    In the population served raster, each pixel is assigned the value corresponding to the total population count in
    a radius of `area_served` meters.

    More specifically, each pixel is assigned the value corresponding to the sum of all its neighboring pixels --
    constrained by a disk-shaped footprint defined by the buffer area.

    Parameters
    ----------
    districts : geodataframe
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

    # Compute population served for each population raster tile, i.e. once per district
    for index, district in districts.iterrows():
        fp = population_dir / f"{index}_served.tif"
        population_raster = population_dir / f"{index}.tif"

        with rasterio.open(str(population_raster)) as src:
            dst_profile = src.profile.copy()

        pop_sum = compute_population_served(population_raster, area_served, district.geometry)

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


# For generate_priority_areas
def already_served(fosa: gpd.GeoDataFrame, min_distance: int, raster_template: Path) -> np.ndarray:
    """Create a mask with areas already served by an existing CSI.

    Parameters
    ----------
    fosa : geodataframe
        A geodataframe with all the CSI.
    min_distance : int
        Min. distance from existing CSI (i.e. buffer radius) in meters.
    raster_template : Path
        Path to a template raster. Output raster will have equal width, height, crs and affine transform.

    Returns
    -------
    ndarray
        Mask with positive values for areas served.
    """
    with rasterio.open(str(raster_template)) as src:
        transform = src.transform
        width, height = src.width, src.height

    return rasterio.features.geometry_mask(
        geometries=[geom.__geo_interface__ for geom in fosa.buffer(min_distance)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )


def generate_priority_areas(
    population_served: Path,
    csi: gpd.GeoDataFrame,
    raster_template: Path,
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
    raster_template : Path
        Path to a template raster. CRS, transform and shape will be used to compute the output raster.
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

    served_by_csi = already_served(fosa=csi, min_distance=min_dist_from_csi, raster_template=raster_template)

    priority_areas[served_by_csi == 1] = -1
    with rasterio.open(str(dst_file), "w", **dst_profile) as dst:
        dst.write(priority_areas, 1)
    return dst_file


def population_served_per_fosa(fosa: gpd.GeoDataFrame, population_served_raster: Path) -> pd.Series:
    """Get population served for each FOSA per district.

    Parameters
    ----------
    fosa : geodataframe
        Geodataframe with FOSAs.
    population_served_raster : Path
        Path to population served raster.

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

    return fosa_pop_served


def analyse_potential_areas(
    priority_areas: Path, csi: gpd.GeoDataFrame, min_population: int = 5000
) -> gpd.GeoDataFrame:
    """Analyse potential areas for extension.

    Parameters
    ----------
    priority_areas : Path
        Path to raster of priority areas.
    csi : geodataframe
        Centres de santé.
    min_population : int
        Min. population served for a potential area.

    Returns
    -------
    geodataframe
        Potential areas with added metrics such as population served
        and distance to nearest CSI.
    """
    with rasterio.open(str(priority_areas)) as src:
        data = src.read(1)
        data[data < min_population] = 0
        data[data >= min_population] = 1
        dst_transform = src.transform
        dst_crs = src.crs

    # Vectorize continuous hot spots in priority areas raster i.e. areas with pixels > min_population
    geoms = []
    for polygon, value in rasterio.features.shapes(data.astype("uint8"), transform=dst_transform):
        if value:
            geoms.append(shape(polygon).simplify(dst_transform.a))

    potential_areas = gpd.GeoDataFrame(index=[i for i in range(0, len(geoms))])
    potential_areas = gpd.GeoDataFrame(geometry=geoms, crs=dst_crs)

    # Compute area in km2
    potential_areas["area"] = round(potential_areas.area * 1e-6, 2)
    potential_areas = potential_areas[potential_areas["area"] >= 1]

    # Get maximum population served in polygon
    stats = zonal_stats([geom for geom in potential_areas.geometry], raster=str(priority_areas), stats=["max"])

    potential_areas["max_population_served"] = [int(s["max"]) for s in stats]
    distance_csi = []
    csi = csi[csi.geometry.is_valid]

    for area in potential_areas.geometry:
        distance_csi.append(round(csi.distance(area.centroid).min() / 1000, 2))

    potential_areas["distance_nearest_csi"] = distance_csi

    return potential_areas


def analyse_cs(
    cs: gpd.GeoDataFrame, csi: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, column: str
) -> gpd.GeoDataFrame:
    """Analyse cases de santé for extension.

    Parameters
    ----------
    cs : geodataframe
        Cases de santé.
    csi : geodataframe
        Centres de santé.
    districts : geodataframe
        Districts.
    column : str
        "population_{int(max_distance_served / 1000)}km"

    Returns
    -------
    geodataframe
        Potential CS with metrics such as population served and distance to
        nearest CSI.
    """
    distance_csi = []
    csi_ = csi[csi.geometry.is_valid]
    for fosa in cs.geometry:
        if fosa:
            distance_csi.append(round(csi_.distance(fosa).min() / 1000, 2))
        else:
            distance_csi.append(None)
    cs["distance_nearest_csi"] = distance_csi
    cs_ = cs[cs.distance_nearest_csi >= 15].copy()

    def _get_pop_in_district(geom: Polygon, districts: gpd.GeoDataFrame) -> float:
        districts_ = districts[districts.contains(geom)]
        if len(districts_) == 0:
            return 0
        return districts_.population_total.to_numpy()[0]

    cs_["population_in_district"] = cs_.geometry.apply(lambda geom: _get_pop_in_district(geom, districts))
    cs_["coverage_impact"] = cs_[column] / cs_["population_in_district"]

    return cs_


def get_files_paths(dirs_path: list[Path], files_path: list[Path]) -> list[Path]:
    """Collect all files contained in the given directories and add the explicitly provided files.

    Parameters
    ----------
    dirs_path : Iterable[Path]
        List of directory paths to scan for files.
    files_path : Iterable[Path]
        List of file paths to include directly.

    Returns
    -------
    List[Path]
        List of all collected file paths.
    """
    collected_files: list[Path] = []

    # Add files coming from directories
    for directory in dirs_path:
        if not directory.exists() or not directory.is_dir():
            continue

        collected_files.extend(p for p in directory.iterdir() if p.is_file())

    # Add explicitly provided files
    for file in files_path:
        if file.exists() and file.is_file():
            collected_files.append(file)

    return collected_files


def split_files_by_district(
    file_paths: list[Path], districts: pd.DataFrame, district_col: str, output_dir: Path
) -> None:
    """Split files by district and save copies filtered by district value.

    Parameters
    ----------
    file_paths : list[Path]
        List of file paths (.csv, .gpkg, .qgz) to process.
    districts : pd.DataFrame
        DataFrame containing district information.
    district_col : str
        Name of the column in districts containing district values.
    output_dir : Path
        Parent directory where district folders will be created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for district in districts[district_col].unique():
        district_folder = output_dir / str(district)
        district_folder.mkdir(exist_ok=True)

        # Remove existing folder if it exists
        if district_folder.exists():
            shutil.rmtree(district_folder)
        district_folder.mkdir()

        # Subfolder for buffer geom
        buffer_folder = district_folder / "buffer_areas"
        buffer_folder.mkdir()

        for fpath in file_paths:
            ext = fpath.suffix.lower()

            # CSV
            if ext == ".csv":
                df = pd.read_csv(fpath)

                if district_col in df.columns:
                    df_filtered = df[df[district_col] == district]
                else:
                    df_filtered = df.copy()
                df_filtered.to_csv(district_folder / fpath.name, index=False)

            # GeoPackage
            elif ext == ".gpkg":
                gdf = gpd.read_file(fpath)

                if district_col in gdf.columns:
                    gdf_filtered = gdf[gdf[district_col] == district]
                else:
                    gdf_filtered = gdf.copy()

                if "_buffer_" in fpath.name:
                    gdf_filtered.to_file(buffer_folder / fpath.name, driver="GPKG")
                else:
                    gdf_filtered.to_file(district_folder / fpath.name, driver="GPKG")

            # QGIS project (qgz) -> just copy, no filter
            elif ext == ".qgz":
                shutil.copy(fpath, district_folder / fpath.name)

            else:
                print(f"Skipping unsupported file type: {fpath}")


def zip_folder(dir_path: Path, output_dir: Path) -> None:
    """Zip all subdirectories contained in a parent directory.

    Parameters
    ----------
    dir_path: Path
        Path to the directory containing subdirectories to zip.
    output_dir: Path
        Directory where the .zip files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in dir_path.iterdir():
        if not subdir.is_dir():
            continue

        normalized_subdir = subdir.name.replace(" ", "_").replace("'", "")
        zip_path = output_dir / normalized_subdir

        # Remove existing zip if it exists
        if zip_path.with_suffix(".zip").exists():
            zip_path.with_suffix(".zip").unlink()

        shutil.make_archive(
            base_name=str(zip_path),
            format="zip",
            root_dir=subdir,
        )


def upload_to_s3(folder_path: str, region: str = "eu-central-1") -> None:
    """Upload all zip files contained in a folder to an S3 bucket.

    Parameters
    ----------
    folder_path : Path
        Path to the directory containing zipped folders (.zip).
    region : str, optional
        S3 region. Defaults to "eu-central-1".
    """
    s3_connection = workspace.s3_connection("s3-carte-sanitaire-niger-public")

    fs = fsspec.filesystem(
        "s3",
        key=s3_connection.access_key_id,
        secret=s3_connection.secret_access_key,
        client_kwargs={"region_name": region},
    )

    for zip_folder in folder_path.iterdir():
        destination = f"health_cov_modelling/distance/full_modelling/{zip_folder.name}"
        s3_path = f"s3://{s3_connection.bucket_name}/{destination}"

        fs.put_file(zip_folder, s3_path, recursive=False)
        print(f"Uploaded {zip_folder} to {s3_path}")
    current_run.log_info("Résultats stockés dans le bucket s3 !")


if __name__ == "__main__":
    extension_coverage()
