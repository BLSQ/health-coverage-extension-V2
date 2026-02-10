import codecs
import os
import shutil
import sys
from typing import List, Sequence, Tuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.merge
import requests
from dhis2 import Api
from fiona import transform
from gooey import Gooey, GooeyParser
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import aligned_target, transform_bounds
from rasterstats import zonal_stats
from shapely.geometry import Point, Polygon, shape
from tqdm import tqdm

if sys.stdout.encoding != "UTF-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
if sys.stderr.encoding != "UTF-8":
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def coverage(
    dhis2_instance: str,
    dhis2_username: str,
    dhis2_password: str,
    districts: str,
    districts_groups: str,
    csi: str,
    csi_groups: str,
    cs: str,
    cs_groups: str,
    write_org_units: bool,
    population: str,
    population_lat: str,
    population_lon: str,
    population_count: str,
    output_dir: str,
    min_population: int,
    min_distance_from_csi: int,
    max_distance_served: int,
    country: str,
    epsg: int,
    un_adj: bool = True,
    constrained: bool = True,
    skip_extension: bool = False,
    overwrite: bool = False,
    show_progress: bool = False,
):
    """Génère les tables et cartes d'extension de la couverture sanitaire."""
    districts = load_districts(
        src_file=districts,
        dhis2_instance=dhis2_instance,
        dhis2_username=dhis2_username,
        dhis2_password=dhis2_password,
        dhis2_groups=districts_groups,
    )

    csi = load_csi(
        src_file=csi,
        dhis2_instance=dhis2_instance,
        dhis2_username=dhis2_username,
        dhis2_password=dhis2_password,
        dhis2_groups=csi_groups,
    )

    cs = load_cs(
        src_file=cs,
        dhis2_instance=dhis2_instance,
        dhis2_username=dhis2_username,
        dhis2_password=dhis2_password,
        dhis2_groups=cs_groups,
    )

    os.makedirs(os.path.join(output_dir, "organisation_units"), exist_ok=True)
    if write_org_units:
        districts.to_file(
            os.path.join(output_dir, "organisation_units", "districts.gpkg")
        )
        csi.to_file(
            os.path.join(output_dir, "organisation_units", "centres_de_sante.gpkg")
        )
        cs.to_file(
            os.path.join(output_dir, "organisation_units", "cases_de_sante.gpkg")
        )

    # copy qgis files
    for fname in (
        "healthcoverage.qgz",
        "healthcoverage.qpt",
        "healthcoverage_atlas.qgz",
    ):
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), "files", fname),
            os.path.join(output_dir, fname),
        )

    population = load_population(
        src_file=population,
        lon_column=population_lon,
        lat_column=population_lat,
        pop_column=population_count,
        country=country,
        un_adj=un_adj,
        constrained=constrained,
        dst_crs=CRS.from_epsg(epsg),
        dst_dir=os.path.join(output_dir, "population"),
        overwrite=overwrite,
    )

    print("Calcule la population totale par district...", flush=True)
    pop_total = count_population_total(boundaries=districts, population=population)
    districts["population_total"] = pop_total

    print("Calcule la couverture sanitaire pour chaque district...", flush=True)
    districts = calculate_population_covered(
        boundaries=districts,
        column_population_count="population_total",
        csi=csi,
        population=population,
        epsg=epsg,
        distances=[5000, 10000, 15000],
    )
    districts.to_crs(f"epsg:{epsg}").to_file(
        os.path.join(output_dir, "population_coverage.gpkg"), driver="GPKG", index=False
    )
    districts.drop(columns=["geometry"]).to_csv(
        os.path.join(output_dir, "population_coverage.csv"), index=False
    )

    if skip_extension:
        print("Modélisation terminée!")
        return

    # remove old directory with population tiles if overwrite = True
    dst_dir = os.path.join(output_dir, "population", "tiles")
    if os.path.isdir(dst_dir) and overwrite:
        shutil.rmtree(dst_dir)

    # re-use existing tiles if directory is not empty and overwrite = False
    if os.path.isdir(dst_dir) and not overwrite:
        if len(os.listdir(dst_dir)) > 0:
            print("Ré-utilise les données de population existantes.", flush=True)
        else:
            # remove empty dir
            shutil.rmtree(dst_dir)

    # generate district tiles if no existing directory is found
    if not os.path.isdir(dst_dir):
        print("Génère les tiles de population pour chaque district...", flush=True)
        os.makedirs(dst_dir, exist_ok=True)
        split_population_raster(
            population, districts, output_dir=dst_dir, show_progress=show_progress
        )

    # write buffer areas to disk
    os.makedirs(os.path.join(output_dir, "buffer_areas"), exist_ok=True)
    for dist in (5, 15):
        # csi
        gpd.GeoDataFrame(
            geometry=csi.to_crs(f"epsg:{epsg}").buffer(dist * 1000)
        ).dissolve().to_file(
            os.path.join(output_dir, "buffer_areas", f"csi_buffer_{dist}km.gpkg"),
            driver="GPKG",
        )

        # cs
        gpd.GeoDataFrame(
            geometry=cs.to_crs(f"epsg:{epsg}").buffer(dist * 1000)
        ).dissolve().to_file(
            os.path.join(output_dir, "buffer_areas", f"cs_buffer_{dist}km.gpkg"),
            driver="GPKG",
        )

    print("Calcule la population desservie...", flush=True)
    dst_file = os.path.join(output_dir, "intermediary", "population_served.tif")
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    served = generate_population_served(
        districts,
        population_dir=dst_dir,
        dst_file=dst_file,
        epsg=epsg,
        area_served=max_distance_served,
        show_progress=show_progress,
    )

    print("Génère les zones d'extension potentielles...", flush=True)
    dst_file = os.path.join(output_dir, "intermediary", "priority_areas.tif")
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    priority_areas = generate_priority_areas(
        population_served=served,
        csi=csi,
        raster_template=served,
        dst_file=dst_file,
        epsg=epsg,
        min_dist_from_csi=min_distance_from_csi,
    )

    print("Calcule la population desservie par chaque CSI...", flush=True)
    column = f"population_{int(max_distance_served / 1000)}km"
    csi[column] = population_served_per_fosa(csi, served)
    csi.to_crs(f"epsg:{epsg}").to_file(
        os.path.join(output_dir, "csi_population_served.gpkg"), driver="GPKG"
    )
    csi.drop(columns=["geometry"]).to_csv(
        os.path.join(output_dir, "csi_population_served.csv"), index=False
    )

    print("Calcule la population desservie par chaque CS...", flush=True)
    cs[column] = population_served_per_fosa(cs, served)
    cs.to_crs(f"epsg:{epsg}").to_file(
        os.path.join(output_dir, "cs_population_served.gpkg"), driver="GPKG"
    )
    cs.drop(columns=["geometry"]).to_csv(
        os.path.join(output_dir, "cs_population_served.csv"), index=False
    )

    print("Analyse les zones potentielles d'extension...", flush=True)
    potential_areas = analyse_potential_areas(priority_areas, csi, epsg, min_population)
    if not potential_areas.empty:
        potential_areas.to_crs(f"epsg:{epsg}").to_file(
            os.path.join(output_dir, "extension_areas.gpkg"), driver="GPKG"
        )

    print("Analyse le potentiel d'extension des CS...", flush=True)
    potential_cs = analyse_cs(cs, csi, districts, column, epsg)
    potential_cs.to_crs(f"epsg:{epsg}").to_file(
        os.path.join(output_dir, "cs_extension_potential.gpkg"), driver="GPKG"
    )
    potential_cs.drop(columns=["geometry"]).to_csv(
        os.path.join(output_dir, "cs_extension_potential.csv"), index=False
    )

    # shutil.rmtree(os.path.join(output_dir, "population_tiles"))
    # shutil.rmtree(os.path.join(output_dir, "worldpop"))

    print("Modélisation terminée !", flush=True)
    return


def org_units_metadata(
    username: str,
    password: str,
    server: str = "http://dhis2.snisniger.ne",
    timeout: int = 60,
) -> pd.DataFrame:
    """Get organisation units metadata from a DHIS2 instance.

    Parameters
    ----------
    username : str
        DHIS2 username.
    password : str
        DHIS2 password.
    server : str, optional
        DHIS2 instance URL.
        dhis2.snis.niger.ne by default.
    timeout : int, optional
        Request timeout in seconds (60s by default).

    Return
    ------
    dataframe
        Org units metadata (one row per org unit).
    """
    api = Api(
        server=server,
        username=username,
        password=password,
        user_agent="blsq/health-coverage-extension",
    )
    r = api.get(
        "metadata",
        params={"organisationUnits": True, "organisationUnitGroups": True},
        timeout=timeout,
    )
    org_units = pd.DataFrame(r.json().get("organisationUnits"))
    org_units.set_index("id", inplace=True, drop=False)
    org_unit_groups = pd.DataFrame(r.json().get("organisationUnitGroups"))
    org_unit_groups.set_index("id", inplace=True, drop=False)
    return org_units, org_unit_groups


def org_units_in_group(org_unit_groups: pd.DataFrame, group_uid: str) -> List[str]:
    """Get list of org units UIDs in a group."""
    if group_uid not in org_unit_groups.index:
        raise ValueError(f"DHIS2 org unit group {group_uid} not found.")
    return [
        ou.get("id") for ou in org_unit_groups.at[group_uid, "organisationUnits"] if ou
    ]


def extract_org_units(
    org_units_meta: pd.DataFrame,
    org_unit_groups_meta: pd.DataFrame,
    groups_included: List[str] = None,
    groups_excluded: List[str] = None,
    levels_included: List[int] = None,
    geom_types: List[str] = ["Point"],
):
    """Extract all org units based on their group and hierarchical lvl.

    Org unit groups can be either whitelisted or blacklisted. Both
    parameters can be used at the same time. Hierarchical levels can
    only be whitelisted.

    Parameters
    ----------
    org_units_meta : pd.DataFrame
        Org units metadata.
    org_unit_groups_meta : pd.DataFrame
        Org unit groups metadata.
    groups_included : list of str, optional
        List of included group UIDs.
    groups_excluded : list of str, optional
        List of excluded group UIDs.
    levels_included : list of int, optional
        List of levels included.
    geom_types : list of str, optional
        Allowed geometry types.

    Return
    ------
    geodataframe
        Extracted org units.
    """
    ou_uids = list(org_units_meta.index)

    if levels_included:
        ou_uids = [
            uid for uid in ou_uids if org_units_meta.at[uid, "level"] in levels_included
        ]

    if groups_included:
        included = []
        for group in groups_included:
            included += org_units_in_group(org_unit_groups_meta, group)
        ou_uids = [uid for uid in ou_uids if uid in included]

    if groups_excluded:
        excluded = []
        for group in groups_excluded:
            excluded += org_units_in_group(org_unit_groups_meta, group)
        ou_uids = [uid for uid in ou_uids if uid not in excluded]

    org_units = org_units_meta.loc[ou_uids]

    # create new columns with full pyramid information

    def _uid_from_path(path: str, lvl: int):
        org_units = [ou for ou in path.split("/") if ou]
        if len(org_units) >= lvl:
            return org_units[lvl - 1]
        return None

    def _name_from_uid(org_units: pd.DataFrame, uid: str):
        if not uid:
            return None
        return org_units.at[uid, "name"]

    for lvl in range(1, 7):
        org_units[f"level_{lvl}_uid"] = org_units.path.apply(
            lambda path: _uid_from_path(path, lvl)
        )

    for lvl in range(1, 7):
        org_units[f"level_{lvl}_name"] = org_units[f"level_{lvl}_uid"].apply(
            lambda uid: _name_from_uid(org_units_meta, uid)
        )

    # to geodataframe
    org_units = org_units[~pd.isna(org_units.geometry)]
    org_units["geom"] = org_units.geometry.apply(shape)
    org_units = gpd.GeoDataFrame(org_units, crs=CRS.from_epsg(4326), geometry="geom")

    # drop irrelevant columns
    columns = [col for col in org_units.columns if col.startswith("level_")]
    columns.append("level")
    columns.append("geom")
    org_units = org_units[columns]

    # rename geometry column
    org_units.rename(columns={"geom": "geometry"}, inplace=True)

    # filter by geom type
    org_units = org_units[np.isin(org_units.geom_type, geom_types)]

    return gpd.GeoDataFrame(org_units.dropna(axis=1, how="all"))


def load_districts(
    src_file: str,
    dhis2_instance: str,
    dhis2_username: str,
    dhis2_password: str,
    dhis2_groups: str,
) -> gpd.GeoDataFrame:
    """Load districts geometries from source file or DHIS2."""
    # From source file
    if src_file:
        districts = gpd.read_file(src_file)
        n_geoms = sum(districts.geometry.is_valid)
        if n_geoms == 0:
            raise ValueError("Le fichier de districts ne contient aucune géométrie.")
        print(f"Districts: {n_geoms} géométries détectées.", flush=True)

    # If not provided and DHIS2 credentials are set, use it
    elif dhis2_instance and dhis2_username and dhis2_password:
        org_units_meta, groups_meta = org_units_metadata(
            username=dhis2_username,
            password=dhis2_password,
            server=dhis2_instance,
            timeout=60,
        )
        included, excluded = _parse_dhis2_groups(dhis2_groups)
        districts = extract_org_units(
            org_units_meta,
            groups_meta,
            groups_included=included,
            groups_excluded=excluded,
            levels_included=None,
            geom_types=["Polygon", "MultiPolygon"],
        )
        print(f"{len(districts)} districts importés depuis DHIS2.", flush=True)
    else:
        raise ValueError(
            "Ni fichier de districts ni crédentiels DHIS2 n'ont été fournis."
        )

    if not districts.crs:
        districts.crs = CRS.from_epsg(4326)

    return districts


def _parse_dhis2_groups(dhis2_groups: str) -> Tuple[List[str], List[str]]:
    """Parse string with DHIS2 groups info.

    Parameters
    ----------
    dhis2_groups : str
        Space-separated list of DHIS2 group UIDs with excluded
        groups prefixed with "-".

    Return
    ------
    list of str
        List of groups to include.
    list of str
        List of groups to exclude.
    """
    include, exclude = [], []
    dhis2_groups = dhis2_groups.strip()
    for group in dhis2_groups.split(" "):
        if group.startswith("-"):
            exclude.append(group[1:])
        else:
            include.append(group)
    return include, exclude


def load_csi(
    src_file: str,
    dhis2_instance: str,
    dhis2_username: str,
    dhis2_password: str,
    dhis2_groups: str,
) -> gpd.GeoDataFrame:
    """Load CSI geometries from source file or DHIS2."""
    # from source file
    if src_file:
        csi = gpd.read_file(src_file)
        n_geoms = sum(csi.geometry.is_valid)
        if n_geoms == 0:
            raise ValueError("Le fichier de CSI ne contient aucune géométrie.")
        print(f"CSI: {n_geoms} géométries détectées.", flush=True)

    # if not provided and DHIS2 credentials are set, extract from DHIS2
    elif dhis2_instance and dhis2_username and dhis2_password and dhis2_groups:
        org_units_meta, groups_meta = org_units_metadata(
            username=dhis2_username,
            password=dhis2_password,
            server=dhis2_instance,
            timeout=60,
        )
        included, excluded = _parse_dhis2_groups(dhis2_groups)
        csi = extract_org_units(
            org_units_meta,
            groups_meta,
            groups_included=included,
            groups_excluded=excluded,
            levels_included=None,
            geom_types=["Point"],
        )
        print(f"{len(csi)} CSI importés depuis DHIS2.", flush=True)

    # raise error if no source file and no DHIS2 credentials
    else:
        raise ValueError("Ni fichier de CSI ni crédentiels DHIS2 n'ont été fournis.")

    if not csi.crs:
        csi.crs = CRS.from_epsg(4326)

    return csi


def load_cs(
    src_file: str,
    dhis2_instance: str,
    dhis2_username: str,
    dhis2_password: str,
    dhis2_groups: str,
) -> gpd.GeoDataFrame:
    """Load cases de santé geometries from source file or DHIS2."""
    # from source file
    if src_file:
        cs = gpd.read_file(src_file)
        n_geoms = sum(cs.geometry.is_valid)
        if n_geoms == 0:
            raise ValueError("Le fichier de CS ne contient aucune géométrie.")
        print(f"CS: {n_geoms} géométries détectées.", flush=True)

    # if not provided and DHIS2 credentials are set, extract from DHIS2
    elif dhis2_instance and dhis2_username and dhis2_password and dhis2_groups:
        org_units_meta, groups_meta = org_units_metadata(
            username=dhis2_username,
            password=dhis2_password,
            server=dhis2_instance,
            timeout=60,
        )
        included, excluded = _parse_dhis2_groups(dhis2_groups)
        cs = extract_org_units(
            org_units_meta,
            groups_meta,
            groups_included=included,
            groups_excluded=excluded,
            levels_included=None,
            geom_types=["Point"],
        )
        print(f"{len(cs)} CS importés depuis DHIS2.", flush=True)

    # raise error if no source file and no DHIS2 credentials
    else:
        raise ValueError("Ni fichier de CSI ni crédentiels DHIS2 n'ont été fournis.")

    if not cs.crs:
        cs.crs = CRS.from_epsg(4326)

    return cs


def load_population(
    # src_file: str,
    # lon_column: str,
    # lat_column: str,
    # pop_column: str,
    country: str,
    un_adj: bool,
    constrained: bool,
    dst_crs: CRS,
    dst_dir: str = None,
    overwrite: bool = False,
) -> str:
    """Load population data.

    Multiple data sources are possible :
        * If a source raster is provided, use it
        * If a source csv or excel file is provided, convert it
          to a raster
        * If no source file is provided, download Worldpop raster

    Parameters
    ----------
    TODO

    Return
    ------
    str
        Path to population raster.
    """
    # if src_file:

    #     extension = os.path.basename(src_file).split(".")[-1].lower()

    #     # population raster is provided by the user
    #     if extension in ("tif", "tiff"):
    #         return src_file

    #     # population excel file is provided by the user
    #     elif extension in ("csv", "xls", "xlsx"):
    #         population = population_from_excel(
    #             fp=src_file,
    #             lon_column=lon_column,
    #             lat_column=lat_column,
    #             pop_column=pop_column,
    #         )
    #         if not population.crs:
    #             population.crs = CRS.from_epsg(4326)
    #         xmin, ymin, xmax, ymax = population.total_bounds
    #         geom = Polygon(
    #             [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
    #         )
    #         population = population.to_crs(dst_crs)
    #         raster = raster_from_excel(
    #             dst_file=os.path.join(dst_dir, "population.tif"),
    #             population=population,
    #             geom=geom,
    #             dst_crs=dst_crs,
    #             xsize=100,
    #             ysize=100,
    #             pop_count_column=pop_column,
    #         )
    #         return raster

    #     # unrecognized file extension
    #     else:
    #         raise ValueError(f"L'extension du fichier {src_file} n'est pas supportée.")

    # no file is provided by the user

    # download population raster from Worldpop
    print("La carte de population n'a pas été renseignée.", flush=True)
    print("Télécharge les données WorldPop...", flush=True)
    dst_dir = os.path.join(dst_dir, "worldpop")
    os.makedirs(dst_dir, exist_ok=True)
    population = download_worldpop(
        country=country,
        output_dir=dst_dir,
        year=2020,
        un_adj=un_adj,
        constrained=constrained,
        show_progress=False,
        overwrite=overwrite,
    )

    return population


# def population_from_excel(
#     fp: str, lon_column: str, lat_column: str, pop_column: str
# ) -> gpd.GeoDataFrame:
#     """Load geodataframe from excel spreadsheet.

#     The user must provide info on which columns contains lat/lon values
#     and population counts.

#     Parameters
#     ----------
#     fp : str
#         Path to excel or CSV file.
#     lon_column : str
#         Column with decimal longitude.
#     lat_column : str
#         Column with decimal latitude.
#     pop_column : str
#         Column with population counts.

#     Return
#     ------
#     geodataframe
#         Output geodataframe with population count per location.
#     """
#     if fp.lower().endswith(".xls") or fp.lower().endswith(".xlsx"):
#         population = pd.read_excel(fp)
#     elif fp.lower().endswith(".csv"):
#         population = pd.read_csv(fp)
#     else:
#         raise ValueError(f"L'extension de {fp} n'est pas supportée.")
#     population["geometry"] = population.apply(
#         lambda row: Point(row[lon_column], row[lat_column]), axis=1
#     )
#     return gpd.GeoDataFrame(population[[pop_column, "geometry"]])


def create_grid(
    geom: Polygon, dst_crs: CRS, xsize: float, ysize: float
) -> Tuple[rasterio.Affine, Tuple[int], Tuple[float]]:
    """Create a raster grid for a given area of interest.

    Parameters
    ----------
    geom : shapely geometry
        Area of interest.
    dst_crs : CRS
        Target CRS as a rasterio CRS object.
    xsize : float
        x spatial resolution.
    ysize : float
        y spatial resolution.

    Returns
    -------
    transform: Affine
        Output affine transform object.
    shape : tuple of int
        Output shape (height, width).
    bounds : tuple of float
        Output bounds.
    """
    bounds = transform_bounds(CRS.from_epsg(4326), dst_crs, *geom.bounds)
    xmin, ymin, xmax, ymax = bounds
    transform = from_origin(xmin, ymax, xsize, ysize)
    ncols = (xmax - xmin) / xsize
    nrows = (ymax - ymin) / ysize
    transform, ncols, nrows = aligned_target(transform, ncols, nrows, (xsize, ysize))
    return transform, (nrows, ncols), bounds


def raster_from_excel(
    dst_file: str,
    population: gpd.GeoDataFrame,
    geom: Polygon,
    dst_crs: CRS,
    xsize: float,
    ysize: float,
    pop_count_column: str,
) -> str:
    """Create a population raster from a population count spreadhseet.

    The values of population counts associated with a given place are
    burned into the corresponding pixels. If multiple places are located
    in the same pixel, population counts are summed.

    Parameters
    ----------
    dst_file : str
        Path to output geotiff raster.
    population : geodataframe
        Population counts.
    geom : shapely polygon
        Area of interest.
    dst_crs : crs object
        Target CRS.
    xsize : float
        Target spatial resolution in dst_crs units.
    ysize : float
        Target spatial resolution in dst_crs units.
    pop_count_column : str
        Name of the column in population dataframe with
        population count.
    """
    dst_transform, dst_shape, dst_bounds = create_grid(
        geom=geom, dst_crs=dst_crs, xsize=xsize, ysize=ysize
    )
    dst_array = rasterize(
        shapes=[
            (loc.geometry.__geo_interface__, loc[pop_count_column])
            for _, loc in population.iterrows()
        ],
        out_shape=dst_shape,
        transform=dst_transform,
        all_touched=False,
        merge_alg=rasterio.features.MergeAlg.add,
        dtype="int32",
    )
    dst_profile = rasterio.profiles.default_gtiff_profile
    dst_profile.update(
        count=1,
        compress="zstd",
        dtype="int32",
        transform=dst_transform,
        height=dst_shape[0],
        width=dst_shape[1],
        nodata=-1,
        crs=dst_crs,
    )
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with rasterio.open(dst_file, "w", **dst_profile) as dst:
        dst.write(dst_array, 1)
    return dst_file


def _build_worldpop_url(country, year, un_adj=True, constrained=True):
    """Build download URL.

    Parameters
    ----------
    country : str
        Country ISO A3 code.
    year : int, optional
        Year of interest (2000--2020).
    un_adj : bool, optional
        Use UN adjusted population counts.
    constrained : bool, optional
        Constrained or unconstrained dataset ?

    Returns
    -------
    url : str
        Download URL.
    """
    if constrained:
        base_url = (
            "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained"
        )
        return (
            f"{base_url}/{year}/maxar_v1/{country.upper()}/"
            f"{country.lower()}_ppp_{year}{'_UNadj' if un_adj else ''}_constrained.tif"
        )
    else:
        base_url = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
        return (
            f"{base_url}/{year}/{country.upper()}/"
            f"{country.lower()}_ppp_{year}{'_UNadj' if un_adj else ''}.tif"
        )


def download_worldpop(
    country,
    output_dir,
    year=2020,
    un_adj=True,
    constrained=True,
    show_progress=True,
    overwrite=False,
):
    """Download a WorldPop population dataset.

    Four types of datasets are supported:
      - Unconstrained (100 m)
      - Unconstrained and UN adjusted (100 m)
      - Constrained (100 m)
      - Constrained and UN adjusted (100 m)

    See Worldpop website for more details:
    <https://www.worldpop.org/project/categories?id=3>

    Parameters
    ----------
    country : str
        Country ISO A3 code.
    output_dir : str
        Path to output directory.
    year : int, optional
        Year of interest (2000--2020). Default=2020.
    un_adj : bool, optional
        Use UN adjusted population counts. Default=False.
    constrained : bool, optional
        Constrained or unconstrained dataset ?
    show_progress : bool, optional
        Show progress bar. Default=False.
    overwrite : bool, optional
        Overwrite existing files. Default=True.

    Return
    ------
    str
        Path to output GeoTIFF file.
    """
    url = _build_worldpop_url(
        country, year=year, un_adj=un_adj, constrained=constrained
    )

    fp = os.path.join(output_dir, url.split("/")[-1])
    if os.path.isfile(fp) and not overwrite:
        print("Données de population déjà téléchargées.", flush=True)
        return fp
    if os.path.isfile(fp) and overwrite:
        os.remove(fp)

    print(f"Téléchargement des données de population depuis {url}.", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()

        size = int(r.headers.get("Content-Length"))

        if show_progress:
            bar_format = "{desc} | {percentage:3.0f}% | {rate_fmt}"
            pbar = tqdm(
                desc=os.path.basename(fp),
                bar_format=bar_format,
                total=size,
                unit_scale=True,
                unit="B",
            )

        with open(fp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    if show_progress:
                        pbar.update(1024)

        if show_progress:
            pbar.close()

    if os.path.getsize(fp) != size:
        os.remove(fp)
        raise ConnectionError("Worldpop download failed and/or is corrupt.")

    print(f"Données de population téléchargées dans {output_dir}.", flush=True)
    return fp


def split_population_raster(
    population_raster: str,
    districts: gpd.GeoDataFrame,
    output_dir: str,
    show_progress: bool = True,
    overwrite: bool = False,
):
    """Split population raster per district.

    The function split the input population raster into
    multiple tiles (one per district). This is to avoid
    taking into account population from other districts
    in the following computations.

    Parameters
    ----------
    population_raster : str
        Path to population raster.
    districts : geodataframe
        Health districts.
    output_dir : str
        Path to output directory.
    show_progress : bool
        Show progress bar.
    overwrite : bool
        Overwrite existing files (default=False).
    """
    if not os.path.isfile(population_raster):
        raise ValueError(f"Population raster not found at {population_raster}.")
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(population_raster) as src:

        # Reproject districts geodataframe if needed
        districts_ = districts.copy()
        if districts_.crs != src.crs:
            districts_ = districts_.to_crs(src.crs)

        dst_profile = src.profile
        dst_profile["compress"] = "zstd"
        dst_profile["predictor"] = 3

        if show_progress:
            pbar = tqdm(total=len(districts_))

        for index, district in districts_.iterrows():

            fp = os.path.join(output_dir, f"{index}.tif")
            if os.path.isfile(fp) and not overwrite:
                if show_progress:
                    pbar.update(1)
                continue

            # Read a window of the population raster based on
            # the district geometry.
            window = rasterio.mask.geometry_window(
                src, shapes=[district.geometry.__geo_interface__]
            )
            transform = src.window_transform(window)
            population = src.read(1, window=window)

            # Make sure that pixels outside the district are assigned
            # the nodata value.
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
            with rasterio.open(
                fp,
                "w",
                **dst_profile,
            ) as dst:
                dst.write(population.astype(np.float32), 1)

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()


def _geom_from_bounds(bounds: Sequence[float]) -> Polygon:
    """Transform bounds tuple into a polygon."""
    xmin, ymin, xmax, ymax = bounds
    return Polygon(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
    )


def get_kernel(population_raster: str, epsg: int, buffer_size: int) -> np.ndarray:
    """Get an array kernel corresponding to a buffer of a given size.

    Generates a flat, disk-shaped footprint as 2d array which
    corresponds to a buffer area around each pixel.

    Parameters
    ----------
    population_raster : str
        Path to population GeoTIFF.
    epsg : int
        EPSG code of the projected coordinate system.
    buffer_size : int
        Buffer size in meters.

    Return
    ------
    kernel : ndarray
        Buffer kernel as a numpy array.
    """
    with rasterio.open(population_raster) as src:

        extent = _geom_from_bounds(src.bounds)
        src_crs = f"EPSG:{src.crs.to_epsg()}"
        dst_crs = f"EPSG:{epsg}"

        # Temporarily reproject the geometry in order to set the buffer radius
        # in meters instead of decimal degrees.
        geom = transform.transform_geom(
            src_crs, dst_crs, extent.centroid.__geo_interface__
        )
        geom = shape(geom).buffer(buffer_size)
        geom = transform.transform_geom(dst_crs, src_crs, geom.__geo_interface__)

        # Rasterize the buffer and use it as a 2d array kernel
        crop, crop_transform = rasterio.mask.mask(
            src, shapes=[geom], crop=True, indexes=1
        )
        kernel = rasterio.mask.geometry_mask(
            [geom], out_shape=crop.shape, transform=crop_transform, invert=True
        )

    return kernel.astype("uint8")


def compute_population_served(
    population_raster: str, epsg: int, area_served: int, geom: Polygon
) -> np.ndarray:
    """Create a raster with population served per pixel.

    In the population served raster, each pixel is assigned
    the value corresponding to the total population count in
    a radius of `area_served` meters.

    More specifically, each pixel is assigned the value corresponding
    to the sum of all its neighboring pixels -- constrained by a
    disk-shaped footprint defined by the buffer area.

    Parameters
    ----------
    population_raster : str
        Path to population GeoTIFF.
    epsg : int
        EPSG code of the projected coordinate system.
    area_served : int
        Area served radius in meters.
    geom : shapely geometry
        Area of interest. Pixel outside will be masked.

    Return
    ------
    ndarray
        Population served per pixel.
    """
    kernel = get_kernel(population_raster, epsg=epsg, buffer_size=area_served)
    with rasterio.open(population_raster) as src:
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


def count_population_total(boundaries: gpd.GeoDataFrame, population: str) -> pd.Series:
    """Count total population per area.

    Parameters
    ----------
    boundaries : geodataframe
        Input districts/areas geometries.
    population : str
        Path to population raster (population per pixel).

    Return
    ------
    series
        Population count per district/area.
    """
    with rasterio.open(population) as src:

        # boundaries geometries must be in same CRS as population raster
        if boundaries.crs != src.crs:
            boundaries_reproj = boundaries.to_crs(src.crs)
        else:
            boundaries_reproj = boundaries.copy()

        stats = zonal_stats(
            raster=src.read(1),
            vectors=boundaries_reproj.geometry,
            stats=["sum"],
            affine=src.transform,
            nodata=src.nodata,
            all_touched=True,
        )

        pop_count = [stat["sum"] for stat in stats]
        return pd.Series(data=pop_count, index=boundaries_reproj.index)


def calculate_population_covered(
    boundaries: gpd.GeoDataFrame,
    column_population_count: str,
    csi: gpd.GeoDataFrame,
    population: str,
    epsg: str,
    distances: List[int] = [5000, 10000, 15000],
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
    population : str
        Path to population raster.
    epsg : int
        EPSG code of coordinate reference system.
    distances : list of int
        Service distances in meters.

    Return
    ------
    dataframe
        Population covered count and ratio as a copy of the boundaries dataframe
        with added columns.
    """
    csi_reproj = csi.to_crs(f"epsg:{epsg}")
    with rasterio.open(population) as src:

        pop = src.read(1)
        coverage = boundaries.copy()
        if coverage.crs != src.crs:
            coverage = coverage.to_crs(src.crs)

        for distance in distances:
            data = []
            covered = csi_reproj.buffer(distance)
            if csi_reproj.crs != src.crs:
                covered = covered.to_crs(src.crs)
            covered = covered.unary_union
            for i, row in coverage.iterrows():
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


def generate_population_served(
    districts: gpd.GeoDataFrame,
    population_dir: str,
    dst_file: str,
    epsg: int,
    area_served: int,
    show_progress: bool = True,
    overwrite: bool = False,
) -> str:
    """Compute population served per pixel.

    In the population served raster, each pixel is assigned
    the value corresponding to the total population count in
    a radius of `area_served` meters.

    More specifically, each pixel is assigned the value corresponding
    to the sum of all its neighboring pixels -- constrained by a
    disk-shaped footprint defined by the buffer area.

    Parameters
    ----------
    districts : geodataframe
        Geodataframe with districts (EPSG:4326).
    population_dir : str
        Path to directory with splitted population rasters.
    dst_file : str
        Path to output file.
    epsg : int
        EPSG code of the projected coordinate system.
    area_served : int
        Area served radius in meters.
    show_progress : bool, optional
        Show a progress bar.
    overwrite : bool, optional
        Overwrite existing files (default=False).

    Return
    ------
    str
        Path to output file.

    Notes
    -----
    The functions uses GDAL, more specifically gdal_merge.py:
    <https://gdal.org/programs/gdal_merge.html>
    """
    if show_progress:
        pbar = tqdm(total=len(districts))

    # get coordinate reference system of population raster and reproject
    # districts geometries if needed
    random_tile = [
        os.path.join(population_dir, f)
        for f in os.listdir(population_dir)
        if f.endswith(".tif")
    ][0]
    with rasterio.open(random_tile) as src:
        population_crs = src.crs
    if districts.crs != population_crs:
        districts.to_crs(population_crs, inplace=True)

    # Compute population served for each population raster tile,
    # i.e. once per district
    for index, district in districts.iterrows():

        # skip if file has already been processed
        fp = os.path.join(population_dir, f"{index}_served.tif")
        if os.path.isfile(fp) and not overwrite:
            if show_progress:
                pbar.update(1)
            continue

        population_raster = os.path.join(population_dir, f"{index}.tif")
        with rasterio.open(population_raster) as src:
            dst_profile = src.profile.copy()
        pop_sum = compute_population_served(
            population_raster, epsg, area_served, district.geometry
        )

        dst_profile["dtype"] = "int32"
        dst_profile["nodata"] = -1
        with rasterio.open(fp, "w", **dst_profile) as dst:
            dst.write(pop_sum, 1)

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    # Merge all population served tiles into a mosaic raster
    if not os.path.isfile(dst_file) or overwrite:

        print("Assemble les tiles de population...", flush=True)
        tiles = [
            os.path.join(population_dir, f)
            for f in os.listdir(population_dir)
            if f.endswith("_served.tif")
        ]

        with rasterio.open(tiles[0]) as src:
            meta = src.meta.copy()

        data, dst_transform = rasterio.merge.merge(tiles)
        meta.update(
            {
                "driver": "GTiff",
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": dst_transform,
            }
        )

        with rasterio.open(dst_file, "w", **meta) as dst:
            dst.write(data)

    return dst_file


def already_served(
    fosa: gpd.GeoDataFrame, crs: str, min_distance: int, raster_template: str
):
    """Create a mask with areas already served by an existing CSI.

    Parameters
    ----------
    fosa : geodataframe
        A geodataframe with all the CSI.
    crs : str
        A projected CRS string used to compute the buffers in meters.
    min_distance : int
        Min. distance from existing CSI (i.e. buffer radius) in meters.
    raster_template : str
        Path to a template raster. Output raster will have equal width, height,
        crs and affine transform.

    Return
    ------
    ndarray
        Mask with positive values for areas served.
    """
    with rasterio.open(raster_template) as src:
        raster_crs = src.crs
        transform = src.transform
        width, height = src.width, src.height

    return rasterio.features.geometry_mask(
        geometries=[
            geom.__geo_interface__
            for geom in fosa.to_crs(crs).buffer(min_distance).to_crs(raster_crs)
        ],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )


def population_served_per_fosa(
    fosa: gpd.GeoDataFrame,
    population_served_raster: str,
) -> pd.Series:
    """Get population served for each FOSA per district.

    Parameters
    ----------
    fosa : geodataframe
        Geodataframe with FOSAs.
    population_served_raster : str
        Path to population served raster.

    Return
    ------
    serie
        Population served for each CSI.
    """
    fosa_pop_served = pd.Series(index=fosa.index, dtype="int32")

    with rasterio.open(population_served_raster) as src:
        src_transform = src.transform
        pop_served = src.read(1)
        src_crs = src.crs

    if fosa.crs != src_crs:
        fosa_ = fosa.to_crs(src_crs)
    else:
        fosa_ = fosa.copy()

    for index, fs in fosa_.iterrows():
        if fs.geometry:
            row, col = rasterio.transform.rowcol(
                src_transform, fs.geometry.x, fs.geometry.y
            )
            try:
                fosa_pop_served.at[index] = int(pop_served[row, col])
            except IndexError:
                fosa_pop_served.at[index] = None
        else:
            fosa_pop_served.at[index] = None

    return fosa_pop_served


def generate_priority_areas(
    population_served: str,
    csi: gpd.GeoDataFrame,
    raster_template: str,
    dst_file: str,
    epsg: int = 32632,
    min_dist_from_csi: int = 15000,
) -> str:
    """Compute a raster showing priority areas.

    Priority areas are locations at more than <buffer_size> of
    an existing CSI and with more than <population_threshold>
    people in a buffer of 5km (as defined in the population_served
    raster).

    Parameters
    ----------
    population_served : str
        Path to population served raster.
    csi : geodataframe
        Geodataframe with CSI geometries.
    raster_template : str
        Path to a template raster. CRS, transform and shape
        will be used to compute the output raster.
    dst_file : str
        Path to output file.
    epsg : int
        EPSG code of the CRS used when computing buffers.
    min_dist_from_csi : int
        Buffer radius in meters. Minimum distance from existing
        CSI for priority areas.

    Return
    ------
    str
        Output file.
    """
    with rasterio.open(population_served) as src:
        priority_areas = src.read(1)
        dst_profile = src.profile.copy()
    served_by_csi = already_served(
        fosa=csi,
        crs=f"EPSG:{epsg}",
        min_distance=min_dist_from_csi,
        raster_template=raster_template,
    )
    priority_areas[served_by_csi == 1] = -1
    with rasterio.open(dst_file, "w", **dst_profile) as dst:
        dst.write(priority_areas, 1)
    return dst_file


def analyse_potential_areas(
    priority_areas: str,
    csi: gpd.GeoDataFrame,
    epsg: int = 32632,
    min_population: int = 5000,
) -> gpd.GeoDataFrame:
    """Analyse potential areas for extension.

    Parameters
    ----------
    priority_areas : str
        Path to raster of priority areas.
    csi : geodataframe
        Centres de santé.
    epsg : int
        EPSG used to compute distances in meters.
    min_population : int
        Min. population served for a potential area.

    Return
    ------
    geodataframe
        Potential areas with added metrics such as population served
        and distance to nearest CSI.
    """
    with rasterio.open(priority_areas) as src:
        data = src.read(1)
        data[data < min_population] = 0
        data[data >= min_population] = 1
        dst_transform = src.transform
        dst_crs = src.crs

    # Vectorize continuous hot spots in priority areas raster
    # i.e. areas with pixels > min_population
    geoms = []
    for polygon, value in rasterio.features.shapes(
        data.astype("uint8"), transform=dst_transform
    ):
        if value:
            geoms.append(shape(polygon).simplify(dst_transform.a))

    potential_areas = gpd.GeoDataFrame(index=[i for i in range(0, len(geoms))])
    potential_areas["geometry"] = geoms
    potential_areas.crs = dst_crs

    # Compute area in km2
    potential_areas["area"] = round(
        potential_areas.to_crs(f"EPSG:{epsg}").area * 1e-6, 2
    )
    potential_areas = potential_areas[potential_areas["area"] >= 1]

    # Get maximum population served in polygon
    stats = zonal_stats(
        [geom for geom in potential_areas.geometry],
        raster=priority_areas,
        stats=["max"],
    )
    potential_areas["max_population_served"] = [int(s["max"]) for s in stats]
    distance_csi = []
    csi_ = csi.to_crs(f"EPSG:{epsg}")
    csi_ = csi_[csi_.geometry.is_valid]
    for area in potential_areas.to_crs(f"EPSG:{epsg}").geometry:
        distance_csi.append(round(csi_.distance(area.centroid).min() / 1000, 2))
    potential_areas["distance_nearest_csi"] = distance_csi

    return potential_areas


def analyse_cs(
    cs: gpd.GeoDataFrame,
    csi: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
    column: str,
    epsg: int = 32632,
) -> gpd.GeoDataFrame:
    """Analyse cases de santé for extension.

    Parameters
    ----------
    cs : geodataframe
        Cases de santé.
    csi : geodataframe
        Centres de santé.
    epsg : int, optional
        EPSG code of the metric CRS.

    Return
    ------
    geodataframe
        Potential CS with metrics such as population served and distance to
        nearest CSI.
    """
    distance_csi = []
    csi_ = csi.to_crs(f"EPSG:{epsg}")
    csi_ = csi_[csi_.geometry.is_valid]
    for fosa in cs.to_crs(f"EPSG:{epsg}").geometry:
        if fosa:
            distance_csi.append(round(csi_.distance(fosa).min() / 1000, 2))
        else:
            distance_csi.append(None)
    cs["distance_nearest_csi"] = distance_csi
    cs = cs[cs.distance_nearest_csi >= 15]

    # add total population in district to CS dataframe
    if districts.crs != cs.crs:
        districts = districts.to_crs(cs.crs)

    def _get_pop_in_district(geom, districts):

        districts_ = districts[districts.contains(geom)]
        if len(districts_) == 0:
            return 0
        return districts_.population_total.values[0]

    cs["population_in_district"] = cs.geometry.apply(
        lambda geom: _get_pop_in_district(geom, districts)
    )
    cs["coverage_impact"] = cs[column] / cs["population_in_district"]

    return cs


@Gooey(
    program_name="Module d'extension de la couverture santé",
    default_size=(800, 600),
    required_cols=1,
    optional_cols=1,
    tabbed_groups=True,
    progress_regex=r"^progress: (\d+)%$",
)
def app():

    parser = GooeyParser(description="Module d'extension de la couverture santé")
    general = parser.add_argument_group(
        "Général",
        (
            "Configurer les paramètres généraux du module d'extension de la "
            "couverture santé.\n\nCes paramètres determinent l'étendue spatiale "
            "de l'analyse (le pays) et le dossier où seront enregistrés "
            "les fichiers de sortie."
        ),
    )
    dhis2 = parser.add_argument_group(
        "DHIS2",
        (
            "Ces paramètres permettent de connecter le module à une instance DHIS2 "
            "afin de télécharger automatiquement les géométries des districts "
            "et la localisation des formations sanitaires.\n\n"
            "Il n'est pas nécessaire de remplir ces champs si les fichiers de districts "
            "et de formations sanitaires sont chargés manuellement par l'utilisateur."
        ),
    )
    fosa = parser.add_argument_group(
        "Formations sanitaires",
        (
            "Identifier les données d'entrée pour les districts et les deux catégories "
            "de formations sanitaires (e.g. centre de santé et case de santé).\n\n"
            "Les données d'entrées peuvent provenir d'un fichier fourni par l'utilisateur "
            "ou directement téléchargées depuis l'instance DHIS2. "
            "Les unités d'organisation à extraire du DHIS2 sont identifiées "
            "par l'identifiant de leur groupe DHIS2. "
            "Plusieurs groupes peuvent être inclus en les séparant par des espaces."
        ),
    )
    population = parser.add_argument_group(
        "Population",
        (
            "Charger les données de population dans le module de couverture santé. "
            "Ignorer l'onglet pour utiliser les données WorldPop. \n\n"
            "Le fichier de population peut être au format raster (GeoTIFF) ou tabulaire (CSV, Excel). "
            "Dans le cas d'un fichier tabulaire, les noms de colonnes avec les données de latitude "
            " et de longitude doivent êtres renseignés par l'utilisateur, tout comme la colonne "
            "contenant les données de dénombrement de la population."
        ),
    )
    worldpop = parser.add_argument_group(
        "Worldpop",
        (
            "Identifier les données de population WorldPop à télécharger. \n\n"
            "La différence entre les différentes cartes disponibles est expliquée "
            "sur le site officiel de WorldPop (https://hub.worldpop.org/project/categories?id=3)."
        ),
    )
    modeling = parser.add_argument_group(
        "Modélisation",
        (
            "Ajuster les paramètres de modélisation du module.\n\n"
            "Ces paramètres permettent de modifier les paramètres utilisés "
            "par le modèle pour calculer les populations desservies et identifier "
            "les formations sanitaires à améliorer."
        ),
    )

    fosa.add_argument(
        "--districts",
        metavar="Districts",
        help="Fichier des districts (Shapefile, Geopackage, ou GeoJSON)",
        widget="FileChooser",
    )

    fosa.add_argument(
        "--districts-groups",
        metavar="Districts (groupes DHIS2)",
        help="Identifiant DHIS2 du groupe d'unités d'organisation à extraire",
        default="CJIuQ1Lp3wG",
    )

    fosa.add_argument(
        "--csi",
        metavar="Centres de santé",
        help="Fichier des centres de santé (Shapefile, Geopackage, ou GeoJSON)",
        widget="FileChooser",
    )
    fosa.add_argument(
        "--csi-groups",
        metavar="Centres de santé (groupes DHIS2)",
        help="Identifiants DHIS2 des groupes d'unités d'organisation à extraire",
        default="iGLtZMdDGMD S6YdxQgX8SO",
    )

    fosa.add_argument(
        "--cs",
        metavar="Cases de santé",
        help="Fichier des cases de santé (Shapefile, Geopackage, ou GeoJSON)",
        widget="FileChooser",
    )
    fosa.add_argument(
        "--cs-groups",
        metavar="Cases de santé (groupes DHIS2)",
        help="Identifiants DHIS2 des groupes d'unités d'organisation à extraire",
        default="EDbDMbIQtPD",
    )

    fosa.add_argument(
        "--write-org-units",
        metavar="Garder une copie des FOSAs",
        help="Enregistre une copie des FOSAs dans le dossier de sortie",
        action="store_true",
        default=False,
    )

    dhis2.add_argument(
        "--dhis2-instance",
        metavar="Instance DHIS2",
        help="URL de l'instance DHIS2",
        default="http://dhis2.snisniger.ne",
    )

    dhis2.add_argument(
        "--dhis2-username", metavar="Utilisateur DHIS2", help="Nom d'utilisateur DHIS2"
    )

    dhis2.add_argument(
        "--dhis2-password",
        metavar="Mot de passe DHIS2",
        help="Mot de passe DHIS2",
        widget="PasswordField",
    )

    modeling.add_argument(
        "--min-distance-csi",
        metavar="Distance minimum",
        type=int,
        help="Distance (en mètres) minimum à un CSI existant pour considérer une nouvelle implantation ou une reconversion.",
        default=15000,
    )

    modeling.add_argument(
        "--max-distance-served",
        metavar="Distance desservie",
        type=int,
        help="Rayon (en mètres) autour d'un CSI autour duquel la population est considérée comme étant desservie.",
        default=5000,
    )

    modeling.add_argument(
        "--min-population",
        metavar="Population desservie minimum",
        type=int,
        help="Population desservie minimum pour qu'une CS soit considérée pour reconversion en CSI.",
        default=5000,
    )

    modeling.add_argument(
        "--skip-extension",
        metavar="Calculer uniquement la couverture sanitaire",
        action="store_true",
        help="Arrêter la modélisation au calcul des statistiques de couverture sanitaire par district (pourcentage de population desservie).",
        default=False,
    )

    general.add_argument(
        "--output-dir",
        metavar="Dossier de sortie",
        help="Dossier où enregistrer les résultats",
        required=True,
        widget="DirChooser",
    )

    general.add_argument(
        "--country", metavar="Pays", help="Code pays (ISO-A3)", type=str, default="NER"
    )

    general.add_argument(
        "--epsg",
        metavar="Système de projection",
        help="Système de projection à utiliser pour la modélisation (EPSG code)",
        type=int,
        default=32632,
    )

    general.add_argument(
        "--overwrite",
        metavar="Overwrite",
        action="store_true",
        help="Forcer la réécriture des données existantes",
        default=False,
    )

    worldpop.add_argument(
        "--un-adj",
        metavar="UN ajustement",
        action="store_true",
        help="Utiliser les données WorldPop ajustées aux prédictions des Nations Unies",
        default=True,
    )

    worldpop.add_argument(
        "--unconstrained",
        metavar="Non-contraint",
        action="store_true",
        help="Utiliser les données WorldPop non-contraintes",
        default=False,
    )

    population.add_argument(
        "--population",
        metavar="Données de population",
        help="Carte de population (GeoTIFF, CSV ou Excel)",
        widget="FileChooser",
    )

    population.add_argument(
        "--population-lat",
        metavar="Latitude",
        help="Nom de la colonne (si CSV ou Excel)",
        type=str,
    )
    population.add_argument(
        "--population-lon",
        metavar="Longitude",
        help="Nom de la colonne (si CSV ou Excel)",
        type=str,
    )
    population.add_argument(
        "--population-count",
        metavar="Dénombrement",
        help="Nom de la colonne (si CSV ou Excel)",
        type=str,
    )

    args = parser.parse_args()

    coverage(
        dhis2_instance=args.dhis2_instance,
        dhis2_username=args.dhis2_username,
        dhis2_password=args.dhis2_password,
        districts=args.districts,
        districts_groups=args.districts_groups,
        csi=args.csi,
        csi_groups=args.csi_groups,
        cs=args.cs,
        cs_groups=args.cs_groups,
        write_org_units=args.write_org_units,
        output_dir=args.output_dir,
        min_population=args.min_population,
        min_distance_from_csi=args.min_distance_csi,
        max_distance_served=args.max_distance_served,
        country=args.country,
        epsg=args.epsg,
        population=args.population,
        population_lat=args.population_lat,
        population_lon=args.population_lon,
        population_count=args.population_count,
        un_adj=args.un_adj,
        constrained=False if args.unconstrained else True,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    app()