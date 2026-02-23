from pathlib import Path

import config
import geopandas as gpd
import utils.geo as geo
from model import Modelling
from openhexa.sdk import current_run, pipeline, workspace
from prepare_outputs import PrepareOutputs


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
    geo_dir = results_dir / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)

    regions_dir = results_dir / "regions"
    districts_dir = results_dir / "districts"

    # Prepare geospatial stuff
    regions = geo.merge_districts(df_shapes=districts, output_dir=geo_dir)

    geo.save_buffered_geom(output_dir=geo_dir, health_facilities=csi, name="csi", buffers=config.buffers)
    geo.save_buffered_geom(output_dir=geo_dir, health_facilities=cs, name="cs", buffers=config.buffers)

    # Modelling
    region = Modelling(
        output_dir=regions_dir,
        boundaries=regions,
        population=population_path,
        csi=csi,
        cs=cs,
        level="region",
    )

    region.population_computing()
    region.extension_computing()

    district = Modelling(
        output_dir=districts_dir,
        boundaries=districts,
        population=population_path,
        csi=csi,
        cs=cs,
        level="district",
    )

    district.population_computing()
    district.extension_computing()

    current_run.log_info("Modélisation terminée ! Préparation du dossier contenant les fichiers de sortie... ")

    region = PrepareOutputs(
        input_dir=regions_dir / "calculs",
        geo_dir=geo_dir,
        output_dir=regions_dir / "final",
        boundaries=regions,
        level="region",
    )

    region.split_files()
    # region.generate_pdf()
    region.generate_upload_folders()

    district = PrepareOutputs(
        input_dir=regions_dir / "calculs",
        geo_dir=geo_dir,
        output_dir=regions_dir / "final",
        boundaries=regions,
        level="district",
    )

    district.split_files()
    district.generate_pdf()
    district.generate_upload_folders()


if __name__ == "__main__":
    extension_coverage()
