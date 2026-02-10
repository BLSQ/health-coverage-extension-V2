import json
from pathlib import Path

import geopandas as gpd
import polars as pl
from openhexa.sdk import current_run, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import (
    get_organisation_unit_levels,
    get_organisation_units,
)
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


@pipeline("dhis2_shapes_extract")
def dhis2_shapes_extract() -> None:
    """Main pipeline function to extract shapes from a DHIS2 instance."""
    current_run.log_info("Shapes pipeline started")

    output_path = Path(workspace.files_path) / "organisation_units" 

    dhis2_client = get_dhis2_client()

    for org_unit_level in ["CQoLZEzDeja", "GvEmxfM7K3k", "z4Qm70cA0k2"]:
        # Country - Region - District
        try:
            retrieve_shapes(dhis2_client, org_unit_level, output_path=output_path)

        except Exception as e:
            current_run.log_error(f"Pipeline error: {e}")


def get_dhis2_client() -> DHIS2:
    """Get the DHIS2 connection.

    Returns
    -------
    DHIS2
        An instance of the DHIS2 client connected to the specified DHIS2 instance.
    """
    try:
        dhis2_client = DHIS2(workspace.dhis2_connection("dhis2-national"), cache_dir=Path(workspace.files_path) / ".cache")
        current_run.log_info(f"Successfully connected to DHIS2 instance {workspace.dhis2_connection("dhis2-national").url}")
        return dhis2_client
    except Exception as e:
        raise Exception(f"Error while connecting to {workspace.dhis2_connection("dhis2-national").url} error: {e}") from e


def retrieve_shapes(dhis2_client: DHIS2, 
                    org_level_id: str, 
                    output_path: Path, 
                    geometry_column: str = "geometry") -> None:
    """Retrieve and save shapes from a DHIS2 instance for a specified organizational level.

    Parameters
    ----------
    dhis2_client: DHIS2
        DHIS2 client to connect to the DB.
    org_level_id : str
        The organizational unit level uid to retrieve shapes for.
    output_path : Path
        The directory where the shapes will be saved.
    geometry_column : str, optional
        The name of the column containing geometry data, by default "geometry".

    Retruns
    -------
    None
    """
    org_levels = get_organisation_unit_levels(dhis2_client)
    org_level = org_levels.filter(pl.col("id") == org_level_id)["level"][0]

    try:
        df_pyramid = get_organisation_units(dhis2_client, max_level=org_level)
        df_pyramid = df_pyramid.filter(pl.col("level") == org_level
                                       ).drop(["id", "name", "level", "opening_date", "closed_date"])
        df_pyramid = df_pyramid.filter(~pl.col("geometry").is_null())

        if len(df_pyramid) == 0:
            raise ValueError("No shapes found for the specified organisation unit level")
        
        shapes = transform_shapes(df_pyramid, geometry_column=geometry_column)
        shapes = shapes[shapes.geom_type != "Point"]
        current_run.log_info(f"{len(shapes)} Shapes extracted for org level {org_level}")

        fname = f"shapes_level{org_level}.gpkg"

        save_shapes(
            shapes=shapes,
            output_path=output_path,
            filename=fname,
        )

        current_run.add_file_output((output_path / fname).as_posix())
    except Exception as e:
        raise Exception(f"Error while extracting shapes: {e}") from e


def transform_shapes(df_shapes: pl.DataFrame, 
                     geometry_column: str = "geometry") -> gpd.GeoDataFrame:
    """Transform a Polars DataFrame containing geometry data into a GeoPandas GeoDataFrame.

    Parameters
    ----------
    df_shapes: pl.DataFrame
        DataFrame containing geometry data
    geometry_column: str
        Column name of the dataframe where geometries are stored

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with geometries converted from GeoJSON strings.
    """

    # Convert GeoJSON strings to Shapely geometries
    def geojson_to_shapely(geojson_str: str) -> BaseGeometry:
        geojson = json.loads(geojson_str)
        return shape(geojson)

    # Apply the conversion to the geometry column
    shapely_geoms = (
        df_shapes[geometry_column]
        .map_elements(geojson_to_shapely, return_dtype=pl.Object)
        .to_list()
    )

    # Convert to GeoPandas DataFrame
    shapes = gpd.GeoDataFrame(
        df_shapes.drop(geometry_column).to_pandas(),
        geometry=shapely_geoms,
        crs="EPSG:4326",  # NOTE: Assuming WGS84 coordinate system
    )

    return shapes.to_crs(32632)


def save_shapes(shapes: gpd.GeoDataFrame, 
                output_path: Path, 
                filename: str) -> None:
    """Save a GeoDataFrame to a file in the specified output path.

    Parameters
    ----------
    shapes : gpd.GeoDataFrame
        The GeoDataFrame containing the shapes to save.
    output_path : Path
        The directory where the file will be saved.
    filename : str
        The name of the file to save the shapes in.
    """
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise Exception(f"Error creating output directory {output_path}: {e}") from e

    try:
        output_fname = Path(output_path).joinpath(filename)
        shapes.to_file(output_fname, driver="GPKG")
        current_run.log_info(f"GeoDataFrame successfully saved to {output_fname}")
    except PermissionError as e:
        raise PermissionError("Error: You don't have permission to access this file.") from e
    except OSError as e:
        raise OSError(f"An I/O error occurred: {e}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e


if __name__ == "__main__":
    dhis2_shapes_extract()