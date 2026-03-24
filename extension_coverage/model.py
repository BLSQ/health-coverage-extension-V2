from dataclasses import dataclass, field
from pathlib import Path

import config
import geopandas as gpd
import utils.analysis as analysis
import utils.geo as geo
import utils.population as pop
from openhexa.sdk import current_run


@dataclass
class Modelling:
    """Compute everything related to population. Then analyse and produce extension areas outputs.

    Attributes
    ----------
    output_dir : Path
        Directory where generated files will be saved.
    boundaries: gpd.GeoDataFrame
        Boundaries of interest (regions or districts)
    population: Path
        Path to the population GeoTIFF raster from WorldPop.
    csi: gpd.GeoDataFrame
        Locations of CSI facilities.
    cs: gpd.GeoDataFrame
        Locations of CS facilities.
    level: str
        Level of interest (regions or districts) (mostly used for log messages)
    distances: list[int] | None = None
        List of distances (in m) used to calculate current health coverage
    max_distance_served:
        Radius within which the population is considered to be served.
    min_distance_from_csi:
        Minimum distance required from any existing CSI for a health zone or box to be considered for extension
    """

    output_dir: Path
    boundaries: gpd.GeoDataFrame
    population: Path
    csi: gpd.GeoDataFrame
    cs: gpd.GeoDataFrame
    level: str
    distances: list[int] = field(default_factory=lambda: config.distances)
    max_distance_served: int = field(default_factory=lambda: config.max_distance_served)
    min_distance_from_csi: int = field(default_factory=lambda: config.min_distance_from_csi)

    def __post_init__(self):
        self.column = f"population_{int(self.max_distance_served / 1000)}km"
        self.boundaries_ = self.boundaries.copy()
        self.csi_ = self.csi.copy()
        self.cs_ = self.cs.copy()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tiles_dir = self.output_dir / "tiles"
        self.calculs_dir = self.output_dir / "calculs"
        self.intermediate_dir = self.output_dir / "intermediate"
        self.population_served = self.intermediate_dir / "population_served.tif"
        self.priority_areas = self.intermediate_dir / "priority_areas.tif"

        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.calculs_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

    def population_computing(self):
        """Run the full population coverage computation workflow.

        This method orchestrates all processing steps required to:
        - Split the population raster by administrative boundaries.
        - Compute total population per administrative unit.
        - Estimate health coverage per unit based on CSI locations and distance thresholds.
        - Generate a raster of population effectively served.
        - Identify potential priority areas for health service extension.
        - Compute the population served by each CSI and CS facility.
        """
        geo.split_raster(
            raster=self.population, boundaries=self.boundaries, level=self.level, output_dir=self.tiles_dir
        )

        current_run.log_info(f"Calcule la population totale par {self.level}...")
        self.boundaries = pop.count_population_total(boundaries=self.boundaries, population=self.population)

        current_run.log_info(f"Calcule la couverture sanitaire pour chaque {self.level}...")
        self.boundaries_ = pop.calculate_population_covered(
            boundaries=self.boundaries,
            column_population_count="population_total",
            csi=self.csi,
            population=self.population,
            distances=config.distances,
            output_dir=self.calculs_dir,
        )

        current_run.log_info("Calcule la population desservie...")
        served = pop.generate_population_served(
            boundaries=self.boundaries_,
            population_dir=self.tiles_dir,
            dst_file=self.population_served,
            area_served=config.max_distance_served,
        )

        current_run.log_info("Génère les zones d'extension potentielles...")
        pop.generate_priority_areas(
            population_served=served,
            csi=self.csi,
            dst_file=self.priority_areas,
            min_dist_from_csi=config.min_distance_from_csi,
        )

        current_run.log_info("Calcule la population desservie par chaque CSI...")
        self.csi_ = pop.population_served_per_fosa(
            fosa=self.csi,
            population_served_raster=served,
            column=self.column,
            output_dir=self.calculs_dir,
            file_name="csi_population_served",
        )

        current_run.log_info("Calcule la population desservie par chaque CS...")
        self.cs_ = pop.population_served_per_fosa(
            fosa=self.cs,
            population_served_raster=served,
            column=self.column,
            output_dir=self.calculs_dir,
            file_name="cs_population_served",
        )

    def extension_computing(self):
        """Perform extension potential analysis based on previously computed coverage outputs.

        This method analyses:
        - Priority areas identified as underserved and meeting minimum population thresholds.
        - The extension potential of CS facilities, based on coverage gaps,
        administrative level, and population served indicators.

        Notes
        -----
        - This method must be executed after `population_computing()`,
        as it depends on generated priority areas and updated facility layers.
        - Minimum population thresholds and analysis parameters are defined
        in the `config` module.
        """
        current_run.log_info("Analyse les zones potentielles d'extension...")
        analysis.analyse_potential_areas(
            priority_areas=self.priority_areas,
            csi=self.csi_,
            min_population=config.min_population,
            output_dir=self.calculs_dir,
        )

        current_run.log_info("Analyse le potentiel d'extension des CS...")
        analysis.analyse_cs(
            cs=self.cs_,
            csi=self.csi_,
            boundaries=self.boundaries_,
            column=self.column,
            level=self.level,
            output_dir=self.calculs_dir,
        )
