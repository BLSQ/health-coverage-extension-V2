import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from openhexa.sdk import current_run, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import (
    get_organisation_unit_groups,
    get_organisation_units,
)
from shapely.geometry import Polygon, shape


@pipeline("dhis2_extract_csi_cs")
def dhis2_extract_csi_cs() -> None:
    """Extract and save CS and CSI geometries from DHIS2.
    
    This pipeline retrieves organisation units and groups from DHIS2,
    assigns priority groups (CS and CSI), aggregates group information,
    filters valid and active health structures, and exports the resulting
    geometries as GeoPackage files.
    """
    current_run.log_info("Extracting org_units and groups")
    df_org, df_grp = get_org_units_and_groups()

    current_run.log_info("Preparing priority groups (CS & CSI)")
    priority_ids = ["EDbDMbIQtPD", "iGLtZMdDGMD"]
    df_org_prep = prepare_priority_groups(df_org, df_grp, priority_ids)

    current_run.log_info("Aggregating all groups")
    df_org_agg = aggregate_all_groups(df_org_prep, df_grp)

    # Rename columns
    df_org_agg = df_org_agg.rename(columns={
                                    # "level_1_id": "id_pays",
                                    # "level_2_id": "id_region",
                                    # "level_3_id": "id_district",
                                    # "level_4_id": "id_commune",
                                    # "level_5_id": "id_aire",
                                    "level_6_id": "level_6_uid",
                                })

    current_run.log_info("Parsing geometries and selection of valid data")
    df_filtered = filter_valid_units(df_org_agg)

    df_clean = trim(df_filtered)

    # Distinct CS and CSI
    cs = gpd.GeoDataFrame(df_clean[df_clean.id_group == "EDbDMbIQtPD"]).set_crs("EPSG:4326").to_crs("EPSG:32632")
    csi = gpd.GeoDataFrame(df_clean[df_clean.id_group == "iGLtZMdDGMD"]).set_crs("EPSG:4326").to_crs("EPSG:32632")

    # Save 
    current_run.log_info("Saving files")
    output_dir = Path(workspace.files_path) / "organisation_units"
    output_dir.mkdir(parents=True, exist_ok=True)

    cs.to_file(output_dir / "CS.gpkg")
    csi.to_file(output_dir / "CSI.gpkg")

    current_run.log_info(f"Files saved in {output_dir}")
    current_run.add_file_output((output_dir / "CS.gpkg").as_posix())
    current_run.add_file_output((output_dir / "CSI.gpkg").as_posix())


def get_org_units_and_groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve organisation units and organisation unit groups from DHIS2.

    This function establishes a connection to the DHIS2 instance configured
    in the workspace, retrieves organisation units and organisation unit
    groups, and returns them as pandas DataFrames.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - the organisation units DataFrame,
        - the organisation unit groups DataFrame.
    """
    try:
        dhis2_client = DHIS2(workspace.dhis2_connection("dhis2-national"), 
                             cache_dir=Path(workspace.files_path) / ".cache")
        current_run.log_info(f"Successfully connected to DHIS2 instance {workspace.dhis2_connection("dhis2-national").url}")
        
    except Exception as e:
        raise Exception(f"Error while connecting to {workspace.dhis2_connection("dhis2-national").url} error: {e}") from e

    unit_groups = get_organisation_unit_groups(dhis2_client).to_pandas()
    org_units = get_organisation_units(dhis2_client).to_pandas()

    return org_units, unit_groups


def prepare_priority_groups(df_org: pd.DataFrame, df_grp: pd.DataFrame, priority_ids: list) -> pd.DataFrame:
    """Assign priority group identifiers and names to organisation units.

    This function selects groups based on a priority order (e.g. CS, CSI),
    resolves cases where an organisation unit belongs to multiple priority
    groups, and assigns the highest-priority group to each organisation unit;
    
    Parameters
    ----------
    df_org : pd.DataFrame
        DataFrame containing organisation units, identified by an ``id`` column.
    df_grp : pd.DataFrame
        DataFrame containing group definitions and their associated
        organisation units.
    priority_ids : list
        Ordered list of group identifiers defining the priority
        (from highest to lowest).

    Returns
    -------
    pd.DataFrame
        Organisation units DataFrame enriched with ``id_group`` and
        ``name_group`` columns corresponding to the selected priority group.
    """
    df_grp_exp = df_grp.explode("organisation_units").rename(
        columns={"id": "id_group", "name": "name_group", "organisation_units": "orgunit_id"}
    )

    df_priority = df_grp_exp[df_grp_exp["id_group"].isin(priority_ids)].copy()
    df_priority["priority"] = df_priority["id_group"].map({g: i for i, g in enumerate(priority_ids)})

    df_priority = (df_priority.sort_values("priority")
                   .drop_duplicates("orgunit_id")
                   [["orgunit_id", "id_group", "name_group"]])

    return df_org.merge(df_priority, left_on="id", right_on="orgunit_id", how="left").drop(columns="orgunit_id")
    

def aggregate_all_groups(df_org: pd.DataFrame, df_grp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all groups associated with each organisation unit.
    
    Parameters
    ----------
    df_org: pd.DataFrame
        DataFrame containing organisation units
    df_grp: pd.DataFrame
        DataFrame containing groups and their associated organisation units

    Returns
    -------
    pd.DataFrame
        Organisation units DataFrame enriched with an ``all_group`` column
        containing the comma-separated list of associated group names.
    """
    df_grp_exp = df_grp.explode("organisation_units").rename(
        columns={"name": "name_group", "organisation_units": "orgunit_id"}
    )

    df_all_groups = (df_grp_exp.groupby("orgunit_id")["name_group"]
                     .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
                     .reset_index(name="all_group"))

    return df_org.merge(df_all_groups, left_on="id", right_on="orgunit_id", how="left").drop(columns="orgunit_id")


def safe_parse_geometry(geom: str | None) -> Polygon | None:
    """Safely parse a JSON geometry string into a Shapely geometry.
    
    Parameters
    ----------
    geom: str
        JSON-encoded geometry string.
    
    Returns
    -------
    Polygon or None
        Parsed Shapely geometry if valid, otherwise ``None``.
    """
    if geom is None or not geom.strip():
        return None
    return shape(json.loads(geom))


def filter_valid_units(df_org: pd.DataFrame) -> pd.DataFrame:
    """Filter valid and active health structures.

    This function removes organisation units without valid geometry or
    group assignment and keeps only structures that are currently open.
    
    Parameter
    ---------
    df_org: pd.DataFrame
        DataFrame containing health structures and related metadata.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only valid and open health structures.
    """
    # Filtering on existing geom & linked to a group (either CS or CSI, not None)
    df_org["geometry"] = df_org["geometry"].apply(safe_parse_geometry)
    df_clean = df_org[df_org["name_group"].notna() & df_org["geometry"].notna()]

    # Filtering on open structures
    df_clean.loc[:, "opening_date"] = pd.to_datetime(df_clean["opening_date"], errors="coerce")
    df_clean.loc[:, "closed_date"] = pd.to_datetime(df_clean["closed_date"], errors="coerce")

    return df_clean[df_clean["closed_date"].isna() | (df_clean["closed_date"] > pd.Timestamp.today())]


def trim(df_org: pd.DataFrame) -> pd.DataFrame:
    """Trim names.
    
    Parameter
    ---------
    df_org: pd.DataFrame
        DataFrame containing health structures.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with trim names.
    """
    cols = [col for col in df_org.columns if "name" in col.lower()]
    df_org[cols] = df_org[cols].apply(lambda x: x.str.strip())
    
    return df_org


if __name__ == "__main__":
    dhis2_extract_csi_cs()