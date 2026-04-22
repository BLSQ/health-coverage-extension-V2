from pathlib import Path

import config
import geopandas as gpd
import utils.geo as geo
from model import CoverageAnalysisPipeline
from openhexa.sdk import current_run, pipeline, workspace
from prepare_outputs import PrepareOutputs


@pipeline("extension_coverage", timeout=21600)
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
    cs = gpd.read_file(org_unit_dir / "CS.gpkg")
    csi = gpd.read_file(org_unit_dir / "CSI.gpkg")
    districts = gpd.read_file(org_unit_dir / "shapes_level3.gpkg")
    population_path = Path(workspace.files_path) / "population/population.tif"

    # Dirs def
    results_dir = Path(workspace.files_path) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    geo_dir = results_dir / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)

    # Prepare geospatial stuff
    regions = geo.merge_districts(df_shapes=districts, output_dir=geo_dir)
    geo.save_buffered_geom(output_dir=geo_dir, health_facilities=csi, name="csi", buffers=config.buffers)
    geo.save_buffered_geom(output_dir=geo_dir, health_facilities=cs, name="cs", buffers=config.buffers)

    # Modelling
    process_level_modelling(
        boundaries=districts,
        geo_dir=geo_dir,
        population=population_path,
        csi=csi,
        cs=cs,
        output_dir=results_dir,
        level="district",
    )
    process_level_modelling(
        boundaries=regions,
        geo_dir=geo_dir,
        population=population_path,
        csi=csi,
        cs=cs,
        output_dir=results_dir,
        level="region",
    )


def process_level_modelling(
    boundaries: gpd.GeoDataFrame,
    geo_dir: Path,
    population: Path,
    csi: gpd.GeoDataFrame,
    cs: gpd.GeoDataFrame,
    output_dir: Path,
    level: str,
):
    """Run the full health coverage modelling pipeline for a given administrative level.

    This function orchestrates the complete workflow for one administrative level:
    1. Initializes and executes the modelling computations (population and extension).
    2. Prepares and restructures the output files.
    3. Splits results, generate folders and upload them to a s3 bucket.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Administrative boundaries for the selected level.
    geo_dir : Path
        Directory containing reference geographic files used during output preparation.
    population : Path
        Path to the population raster file.
    csi : gpd.GeoDataFrame
        GeoDataFrame containing CSI health facility locations.
    cs : gpd.GeoDataFrame
        GeoDataFrame containing CS health facility locations.
    output_dir : Path
        Root directory where level-specific outputs will be stored.
    level : str
        Administrative level identifier (e.g., "region", "district").

    Notes
    -----
    This function acts as a high-level orchestrator and delegates computation and output
    generation to the `CoverageAnalysisPipeline` and `PrepareOutputs` classes.
    """
    dst_dir = output_dir / level

    model = CoverageAnalysisPipeline(
        output_dir=dst_dir,
        boundaries=boundaries,
        population=population,
        csi=csi,
        cs=cs,
        level=level,
    )

    model.population_computing()
    model.extension_computing()

    current_run.log_info(
        f"Modélisation terminée au niveau {level}! Préparation du dossier contenant les fichiers de sortie... "
    )

    outputs = PrepareOutputs(
        input_dir=dst_dir / "calculs",
        geo_dir=geo_dir,
        output_dir=dst_dir / "final",
        boundaries=boundaries,
        level=level,
    )

    outputs.split_files()
    outputs.generate_pdf()
    outputs.generate_upload_folders()


if __name__ == "__main__":
    extension_coverage()
