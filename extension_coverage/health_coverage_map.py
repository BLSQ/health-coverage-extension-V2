from dataclasses import dataclass
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import box


@dataclass
class BaseHealthCoverageMap:

    output_dir: Path
    population_coverage: gpd.GeoDataFrame
    csi_population_served: gpd.GeoDataFrame
    cs_population_served: gpd.GeoDataFrame
    cs_extension_potential: gpd.GeoDataFrame
    csi_buffer_5km: gpd.GeoDataFrame
    csi_buffer_15km: gpd.GeoDataFrame
    extension_areas: gpd.GeoDataFrame
    country: gpd.GeoDataFrame
    table_is_empty: bool = None

    def clip_datasets_to_population(self) -> None:
        """Clip all relevant datasets to population coverage."""
        self.csi_population_served = gpd.clip(self.csi_population_served, self.population_coverage)
        self.cs_population_served = gpd.clip(self.cs_population_served, self.population_coverage)
        self.cs_extension_potential = gpd.clip(self.cs_extension_potential, self.population_coverage)
        self.csi_buffer_5km = gpd.clip(self.csi_buffer_5km, self.population_coverage)
        self.csi_buffer_15km = gpd.clip(self.csi_buffer_15km, self.population_coverage)
        self.extension_areas = gpd.clip(self.extension_areas, self.population_coverage)

    def prepare_extension_table(self) -> tuple[gpd.GeoDataFrame | pd.DataFrame, bool]:
        """Prepare summary table for CS extension potential.
        
        Returns
        -------
        table: gpd.GeoDataFrame | pd.DataFrame
            Output table which will be displayed in the pdf.
        table_is_empty: bool
            Booleen used later for formatting purposes.
        """
        cols = ["id", "name", "population_5km", "coverage_impact", "distance_nearest_csi"]

        if self.cs_extension_potential.empty:
            table = pd.DataFrame([["", "", "Aucune CS à impact élevé n'a été identifiée.", "", ""]],
                                 columns=cols)
            self.table_is_empty = True
        else:
            table = self.cs_extension_potential[cols][:5].copy().sort_values("coverage_impact", ascending=False)
            table["population_5km"] = "+ " + table["population_5km"].astype(int).astype(str)
            table["coverage_impact"] = "+ " + (100 * table["coverage_impact"]).round(1).astype(str) + "%"
            table["distance_nearest_csi"] = table["distance_nearest_csi"].round(1).astype(str) + " km"
            self.table_is_empty = False

        return table, self.table_is_empty

    def create_figure_layout(self) -> tuple[Figure, Axes, Axes]:
        """Create matplotlib figure with map + table layout.
        
        Returns
        -------
        fig: Figure
        ax_map: Axes
        ax_table: Axes
        """
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[0.01, 6, 1.7 - self.table_is_empty])
        ax_blank = fig.add_subplot(gs[0])
        ax_blank.axis("off")
        ax_map = fig.add_subplot(gs[1])
        ax_table = fig.add_subplot(gs[2])

        return fig, ax_map, ax_table

    def compute_map_masks(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Compute outside-country and outside-district masks.

        Returns
        -------
        outside_country: gpd.GeoDataFrame
            Output dataframe whose geometry is outside the country.
        outside_popcov: gpd.GeoDataFrame
            Output dataframe whose geometry is outside the district.
        """
        minx, miny, maxx, maxy = self.country.total_bounds
        padding = 0.2
        big_extent = box(
            minx - padding * (maxx - minx),
            miny - padding * (maxy - miny),
            maxx + padding * (maxx - minx),
            maxy + padding * (maxy - miny),
        )
        extent_gdf = gpd.GeoDataFrame(geometry=[big_extent], crs=self.country.crs)
        country_geom = self.country.union_all()
        popcov_geom = self.population_coverage.union_all()
        country_gdf = gpd.GeoDataFrame(geometry=[country_geom], crs=self.country.crs)
        popcov_gdf = gpd.GeoDataFrame(geometry=[popcov_geom], crs=self.country.crs)
        outside_country = gpd.overlay(extent_gdf, country_gdf, how="difference")
        outside_popcov = gpd.overlay(country_gdf, popcov_gdf, how="difference")

        return outside_country, outside_popcov

    def export_pdf(self, fig: Figure, filename: str = "health_coverage_map.pdf") -> Path:
        """Export figure as pdf.
    
        Returns
        -------
        output_path: Path
            Path of the generated pdf
        """
        output_path = self.output_dir / filename
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return output_path


@dataclass
class DistrictHealthCoverageMap(BaseHealthCoverageMap):

    def generate(self) -> Path:
        """Generate a pdf file for given data.

        Returns
        -------
        self._export_pdf(fig): Path
            Path to the generated pdf.
        """
        self.clip_datasets_to_population()
        
        table, self.table_is_empty = self.prepare_extension_table()

        fig, ax_map, ax_table = self.create_figure_layout()

        outside_country, outside_popcov = self.compute_map_masks()
        self.plot_map_layers(ax_map, outside_country, outside_popcov)
        self.plot_table(ax_table, table)

        return self.export_pdf(fig)
    
    def plot_map_layers(self, ax: Axes, outside_country: gpd.GeoDataFrame, outside_popcov: gpd.GeoDataFrame):
        """Draw all map layers."""
        # zoom
        minx, miny, maxx, maxy = self.population_coverage.total_bounds
        pad_x = (maxx - minx) * 0.1
        pad_y = (maxy - miny) * 0.1
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        self.population_coverage.plot(ax=ax, alpha=0)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # buffers
        self.csi_buffer_15km.plot(ax=ax, color="#b2df8a", alpha=0.25, edgecolor="#33a02c")
        self.csi_buffer_5km.plot(ax=ax, color="#b2df8a", alpha=0.25, edgecolor="#33a02c")

        # extension areas
        self.extension_areas.plot(ax=ax, color="#ff0004", edgecolor="#a4181a")
        for x, y, pop in zip(self.extension_areas.geometry.centroid.x,
                             self.extension_areas.geometry.centroid.y,
                             self.extension_areas.max_population_served, 
                             strict=False):
            label = f"+{pop}"
            text = ax.annotate(label, xy=(x, y), fontsize=9, color="#610023")
            text.set_path_effects([pe.withStroke(linewidth=2, foreground="#fafafa")])

        # points
        self.csi_population_served.plot(ax=ax, color="#cc0000", markersize=50, edgecolor="#000000")
        for x, y, label in zip(self.csi_population_served.geometry.x,
                               self.csi_population_served.geometry.y,
                               self.csi_population_served.name, 
                               strict=False):
            text = ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=9, weight="bold")
            text.set_path_effects([pe.withStroke(linewidth=2, foreground="#fafafa")])

        self.cs_population_served.plot(ax=ax, color="#0077fe", markersize=50, edgecolor="#000000")

        self.cs_extension_potential.plot(ax=ax, color="#54b252", markersize=50, edgecolor="#000000")
        for x, y, name, pop in zip(self.cs_extension_potential.geometry.x, 
                                   self.cs_extension_potential.geometry.y,
                                   self.cs_extension_potential.name, 
                                   self.cs_extension_potential.population_5km, 
                                   strict=False):
            label = f"{name}\n+{int(pop)}"
            text = ax.annotate(label, xy=(x, y), xytext=(-3, -3), textcoords="offset points", 
                               fontsize=9, ha="right", va="top")
            text.set_path_effects([pe.withStroke(linewidth=2, foreground="#fafafa")])

        # contours
        self.country.boundary.plot(ax=ax, color="black", linewidth=1)
        self.population_coverage.boundary.plot(ax=ax, color="black", linewidth=1)

        outside_country.plot(ax=ax, facecolor="none", edgecolor="lightgrey", hatch="////", linewidth=0)
        outside_popcov.plot(ax=ax, color="white", alpha=0.7)
        
        ax.set_title(f"{self.population_coverage.level_3_name[0]}, {self.population_coverage.level_2_name[0]}", 
                     fontsize=20, fontweight="bold", pad=90)
        ax.text(0.5, 1.1, f"Population calculée : {int(self.population_coverage.population_total)}", 
                transform=ax.transAxes, ha="center", va="bottom", fontsize=15)
        ax.axis("off")

        # legend
        legend_elements = [
            Patch(facecolor="#b2df8a", edgecolor="#33a02c", alpha=0.75, label="Zone desservie (5 km)"),
            Patch(facecolor="#b2df8a", edgecolor="#33a02c", alpha=0.45, label="Zone desservie (15 km)"),
            Patch(facecolor="#ff0004", edgecolor="#a4181a", label="Zone d'implantation potentielle"),
            Line2D([0], [0], marker="o", color="#cc0000", markerfacecolor="#cc0000", markersize=7, ls="", 
                   markeredgecolor="#000000", label="CSI existant"),
            Line2D([0], [0], marker="o", color="#cc0000", markerfacecolor="#54b252", markersize=7, ls="", 
                   markeredgecolor="#000000", label="CS (impact élevé)"),
            Line2D([0], [0], marker="o", color="#cc0000", markerfacecolor="#0077fe", markersize=7, ls="", 
                   markeredgecolor="#000000", label="CS (impact faible)"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=8, frameon=True)
        
        # scale
        scalebar = ScaleBar(dx=1, units="m", dimension="si-length", length_fraction=0.25, 
                            location="lower right", scale_loc="top", pad=0.5, color="#343837", 
                            box_color="white", box_alpha=1, font_properties={"size": 10})
        ax.add_artist(scalebar)
        plt.tight_layout()

    def plot_table(self, ax_table: Axes, table: gpd.GeoDataFrame | pd.DataFrame):
        """Render summary table under map."""
        ax_table.axis("off")
        ax_table.text(
            0.5, 1.4 + self.table_is_empty * 0.2,
            "Couverture sanitaire actuelle calculée (vol d'oiseau)",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold"
        )

        ax_table.text(
            0.5, 1.2,
            f"5 km : {int(100 * self.population_coverage.population_covered_5km_ratio[0])}%"
            f"   10 km : {int(100 * self.population_coverage.population_covered_10km_ratio[0])}%"
            f"   15 km : {int(100 * self.population_coverage.population_covered_15km_ratio[0])}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

        table = ax_table.table(
            cellText=table.values,
            colLabels=["DHIS2 UID", "Nom", "Population desservie", "Impact", "Distance CSI"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            bbox=[0.05, 0.1, 0.9, 0.9]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)

        for (row, _), cell in table.get_celld().items():
        
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_height(cell.get_height() * 1.3)
        
            cell.visible_edges = "horizontal"
            cell.set_edgecolor("#bdbdbd")
            cell.set_linewidth(0.6)