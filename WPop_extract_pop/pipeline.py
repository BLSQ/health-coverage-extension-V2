from pathlib import Path
from datetime import datetime
from osgeo import gdal

from openhexa.sdk import current_run, pipeline, workspace
from worldpopclient import WorldPopClient


@pipeline("wpop_extract_population")
def wpop_extract_population():
    """Pipeline to extract data from WorldPop and reproject it in EPSG:32632."""

    # set paths
    root_path = Path(workspace.files_path)
    output_path = root_path / "population"

    current_year = datetime.now().year
    year = str(2030 if current_year > 2030 else current_year)

    try:
        pop_file_path = retrieve_population_data(
            year=year,
            output_path=output_path
        )

        pop_repro_file_path = reproject(
            tif_file_path=pop_file_path,
            output_dir=output_path
        )

        current_run.add_file_output(pop_repro_file_path.as_posix())

    except Exception as e:
        current_run.log_error(f"Error : {e}")
        raise


def retrieve_population_data(output_path: Path, year: str) -> Path:
    """Retrieve raster population data from worldpop.

    Parameters
    ----------
    output_path : Path
        The directory where the population data will be saved.
    year : str, optional
        The year for which to retrieve the population data. Defaults to "2020".

    Returns
    -------
    Path
        The path to the downloaded population data file.

    """
    current_run.log_info(f"Retrieving population data grid from Worldpop for year {year}.")
    wpop_client = WorldPopClient()
    current_run.log_info(f"Downloading data from : {wpop_client.base_url}")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    # pop_filename = wpop_client.target_tif_filename(year)  # cleaner solution
    pop_file_path = output_path / "population_raw.tif"
    current_run.log_info(f"Retrieving data for Niger - year: {year}")

    try:
        if pop_file_path.exists():
            current_run.log_info(f"File {pop_file_path} already exists. Skipping download")
            return pop_file_path

        wpop_client.download_data_for_country(
            year=year,
            output_dir=output_path,
        )
        current_run.log_info(f"Population data successfully downloaded under : {pop_file_path}")
        return pop_file_path

    except Exception as e:
        raise Exception(
            f"Error retrieving WorldPop population data for Niger - {year}: {e}"
        ) from e


def reproject(tif_file_path=Path, output_dir=Path):
    """Reproject raster population data from EPSG:4326 to EPSG:32632.

    Parameters
    ----------
    tif_file_path: Path
        The path to the downloaded population data file.
    output_dir : Path
        The directory where the reprojected population data will be saved.

    Returns
    -------
    Path
        The path to the reprojected population data file.

    """
    
    output_raster = output_dir / f"population{tif_file_path.suffix}"
    current_run.log_info("Reprojecting population raster from EPSG:4326 to EPSG:32632.")

    try:
        if output_raster.exists():
            current_run.log_info(f"File {output_raster} already exists. Skipping reprojection")
            return output_raster

        ds = gdal.Open(str(tif_file_path))
        if ds is None:
            raise RuntimeError(f"Cannot open population raster: {tif_file_path}")
        
        warp_options = gdal.WarpOptions(
                                        srcSRS="EPSG:4326",
                                        dstSRS="EPSG:32632",
                                        resampleAlg="near",
                                        format="GTiff",
                                        creationOptions=["TILED=YES", "COMPRESS=DEFLATE"],
                                        multithread=True
                                        )

        result = gdal.Warp(
                            destNameOrDestDS=str(output_raster),
                            srcDSOrSrcDSTab=ds,
                            options=warp_options
                        )

        if result is None:
            raise RuntimeError(f"Reprojection failed for {tif_file_path}")

        ds = None
        result = None

        current_run.log_info(f"Population successfully reprojected under : {output_raster}")

        return output_raster

    except Exception as e:
        raise Exception(
            f"Error reprojecting population raster: {e}"
        ) from e


if __name__ == "__main__":
    wpop_extract_population()