from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Polygon, shape


def analyse_potential_areas(
    priority_areas: Path, csi: gpd.GeoDataFrame, output_dir: Path, min_population: int = 5000
) -> gpd.GeoDataFrame:
    """Analyse potential areas for extension.

    Parameters
    ----------
    priority_areas : Path
        Path to raster of priority areas.
    csi : geodataframe
        Centres de santé.
    output_dir: Path
        Output directory path where file will be stored
    min_population : int
        Min. population served for a potential area.

    Returns
    -------
    geodataframe
        Potential areas with added metrics such as population served and distance to nearest CSI.
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

    if not potential_areas.empty:
        potential_areas.to_file(output_dir / "extension_areas.gpkg", driver="GPKG")

    return potential_areas


def analyse_cs(
    cs: gpd.GeoDataFrame, csi: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame, column: str, level: str, output_dir: Path
):
    """Analyse cases de santé for extension.

    Parameters
    ----------
    cs : geodataframe
        Cases de santé.
    csi : geodataframe
        Centres de santé.
    boundaries : geodataframe
        Areas.
    column : str
        "population_{int(max_distance_served / 1000)}km"
    level : str
        Either region of district.
    output_dir: Path
        Output directory path.
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

    def _get_pop_in_boundary(geom: Polygon, boundaries: gpd.GeoDataFrame) -> float:
        boundaries_ = boundaries[boundaries.contains(geom)]
        if len(boundaries_) == 0:
            return 0
        return boundaries_.population_total.to_numpy()[0]

    cs_[f"population_in_{level}"] = cs_.geometry.apply(lambda geom: _get_pop_in_boundary(geom, boundaries))
    cs_["coverage_impact"] = cs_[column] / cs_[f"population_in_{level}"]

    cs_.to_file(output_dir / "cs_extension_potential.gpkg", driver="GPKG")
    pd.DataFrame(cs_.drop(columns=["geometry"])).to_csv(output_dir / "cs_extension_potential.csv", index=False)
