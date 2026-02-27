from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import utils.outputs_dealing as od
from health_coverage_map import DistrictHealthCoverageMap, RegionHealthCoverageMap
from openhexa.sdk import current_run, workspace


@dataclass
class PrepareOutputs:
    """Prepare outputs to be uploaded into s3 bucket.

    Attributes
    ----------
    input_dir: Path
        Path to the directory were calcul results are stored.
    geo_dir: Path
        Path to the directory containing geospatial files (buffered geom & country and regions boundaries).
    output_dir : Path
        Directory where generated files will be saved.
    boundaries: gpd.GeoDataFrame
        Boundaries of interest (regions or districts)
    level: str
        Level of interest (regions or districts) (mostly used for log messages)
    """

    input_dir: Path
    geo_dir: Path
    output_dir: Path
    boundaries: gpd.GeoDataFrame
    level: str

    def __post_init__(self):
        self.unzip = self.output_dir / "unzip"
        self.zip = self.output_dir / "zip"
        self.zone_col = "level_3_name" if self.level == "district" else "level_2_name"

        self.unzip.mkdir(parents=True, exist_ok=True)
        self.zip.mkdir(parents=True, exist_ok=True)

    def split_files(self):
        """Split files per zone name."""
        files_paths = od.get_files_paths(
            dirs_path=[self.geo_dir, self.input_dir],
            files_path=[Path(workspace.files_path) / "healthcoverage_atlas.qgz"],
        )

        current_run.log_info(f"Divise les fichiers par {self.level}")
        od.split_files_by_zone(
            file_paths=files_paths, df=self.boundaries, zone_col=self.zone_col, output_dir=self.unzip
        )

    def generate_pdf(self):
        """Generate a pdf with a map of the zone and the first 5 CS to upgrade as CSI."""
        current_run.log_info(f"Génère le PDF atlas pour chaque {self.level}...")

        for folder in self.unzip.iterdir():
            if not folder.is_dir():
                continue

            zone_name = folder.name

            population_coverage = gpd.read_file(folder / "population_coverage.gpkg")
            cs_population_served = gpd.read_file(folder / "cs_population_served.gpkg")
            csi_population_served = gpd.read_file(folder / "csi_population_served.gpkg")
            cs_extension_potential = gpd.read_file(folder / "cs_extension_potential.gpkg")
            extension_areas = gpd.read_file(folder / "extension_areas.gpkg")
            country_gpkg = gpd.read_file(folder / "country.gpkg")
            buffer_dir = folder / "buffer_areas"
            csi_buffer_5km = gpd.read_file(buffer_dir / "csi_buffer_5km.gpkg")
            csi_buffer_15km = gpd.read_file(buffer_dir / "csi_buffer_15km.gpkg")

            if self.level == "district":
                district_map = DistrictHealthCoverageMap(
                    output_dir=folder,
                    population_coverage=population_coverage,
                    csi_population_served=csi_population_served,
                    cs_population_served=cs_population_served,
                    cs_extension_potential=cs_extension_potential,
                    extension_areas=extension_areas,
                    csi_buffer_5km=csi_buffer_5km,
                    csi_buffer_15km=csi_buffer_15km,
                    country=country_gpkg,
                    zone_name=zone_name,
                )

                pdf_path = district_map.generate()
                current_run.log_info(f"PDF généré pour {zone_name} : {pdf_path.name}")

            else:
                region_map = RegionHealthCoverageMap(
                    output_dir=folder,
                    population_coverage=population_coverage,
                    csi_population_served=csi_population_served,
                    cs_population_served=cs_population_served,
                    cs_extension_potential=cs_extension_potential,
                    extension_areas=extension_areas,
                    csi_buffer_5km=csi_buffer_5km,
                    csi_buffer_15km=csi_buffer_15km,
                    country=country_gpkg,
                    zone_name=zone_name,
                )
                pdf_path = region_map.generate()
                current_run.log_info(f"PDF généré pour {zone_name} : {pdf_path.name}")

    def generate_upload_folders(self):
        """Zip and upload to s3 bucket to make docs accessible through the interface cartesanitaireniger.org."""
        od.zip_folder(dir_path=self.unzip, output_dir=self.zip)
        od.upload_to_s3(folder_path=self.zip)  # NOTE: verifier chemin d'accès + chemin ds s3 !!!!
