import shutil
from pathlib import Path

import fsspec
import geopandas as gpd
import pandas as pd
from openhexa.sdk import current_run, workspace


def get_files_paths(dirs_path: list[Path], files_path: list[Path]) -> list[Path]:
    """Collect all files contained in the given directories and add the explicitly provided files.

    Parameters
    ----------
    dirs_path : Iterable[Path]
        List of directory paths to scan for files.
    files_path : Iterable[Path]
        List of file paths to include directly.

    Returns
    -------
    List[Path]
        List of all collected file paths.
    """
    collected_files: list[Path] = []

    # Add files coming from directories
    for directory in dirs_path:
        if not directory.exists() or not directory.is_dir():
            continue

        collected_files.extend(p for p in directory.iterdir() if p.is_file())

    # Add explicitly provided files
    for file in files_path:
        if file.exists() and file.is_file():
            collected_files.append(file)

    return collected_files


def split_files_by_zone(file_paths: list[Path], df: pd.DataFrame, zone_col: str, output_dir: Path) -> None:
    """Split files by district and save copies filtered by district value.

    Parameters
    ----------
    file_paths : list[Path]
        List of file paths (.csv, .gpkg, .qgz) to process.
    df : pd.DataFrame
        DataFrame containing zone information.
    zone_col : str
        Name of the column in districts containing district values.
    output_dir : Path
        Parent directory where district folders will be created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for zone in df[zone_col].unique():
        zone_folder = output_dir / str(zone)

        # Remove existing folder if it exists
        if zone_folder.exists():
            shutil.rmtree(zone_folder)
        zone_folder.mkdir()

        # Subfolder for buffer geom
        buffer_folder = zone_folder / "buffer_areas"
        buffer_folder.mkdir()

        for fpath in file_paths:
            if fpath.name == "regions.gpkg" and zone_col == "level_3_name":
                continue

            ext = fpath.suffix.lower()

            # CSV
            if ext == ".csv":
                df = pd.read_csv(fpath)

                if zone_col in df.columns:
                    df_filtered = df[df[zone_col] == zone]
                else:
                    df_filtered = df.copy()
                df_filtered.to_csv(zone_folder / fpath.name, index=False)

            # GeoPackage
            elif ext == ".gpkg":
                gdf = gpd.read_file(fpath)

                if zone_col in gdf.columns:
                    gdf_filtered = gdf[gdf[zone_col] == zone]
                else:
                    gdf_filtered = gdf.copy()

                if "_buffer_" in fpath.name:
                    gdf_filtered.to_file(buffer_folder / fpath.name, driver="GPKG")
                else:
                    gdf_filtered.to_file(zone_folder / fpath.name, driver="GPKG")

            # QGIS project (qgz) -> just copy, no filter
            elif ext == ".qgz":
                shutil.copy(fpath, zone_folder / fpath.name)

            else:
                print(f"Skipping unsupported file type: {fpath}")


def create_region_pdf_folders(
    region_district_map: dict[str, list[str]],
    regions_path: Path,
    districts_path: Path,
    output_path: Path,
) -> None:
    """Create one folder per region containing the region PDF and the PDFs of its associated districts.

    Parameters
    ----------
    region_district_map : dict
        Dictionary mapping region names to list of district names.
    regions_path : Path
        Path containing region folders.
    districts_path : Path
        Path containing district folders.
    output_path : Path
        Path where the region PDF folders will be created.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    for region, districts in region_district_map.items():
        normalized_region = region.replace(" ", "_").replace("'", "")
        region_folder = regions_path / region

        if not region_folder.exists():
            print(f"Region folder not found: {region_folder}")
            continue

        # Create output folder: region_PDF
        region_output_folder = output_path / f"{normalized_region}_PDF"
        region_output_folder.mkdir(exist_ok=True)

        # Copy region PDF
        region_pdfs = list(region_folder.glob("*.pdf"))

        if not region_pdfs:
            print(f"No PDF found in region folder {region_folder}")
        else:
            for pdf in region_pdfs:
                shutil.copy2(pdf, region_output_folder / pdf.name)

        # Copy district PDF
        for district in districts:
            district_folder = districts_path / district

            if not district_folder.exists():
                print(f"District folder not found: {district_folder}")
                continue

            district_pdfs = list(district_folder.glob("*.pdf"))

            if not district_pdfs:
                print(f"No PDF found in {district_folder}")
                continue

            for pdf in district_pdfs:
                shutil.copy2(pdf, region_output_folder / pdf.name)

        print(f"Created PDF folder for region: {region}")


def copy_districts_into_regions(
    region_district_map: dict[str, list[str]],
    regions_path: Path,
    districts_path: Path,
) -> None:
    """Copy district folders into their corresponding region folders.

    Parameters
    ----------
    region_district_map : dict
        Dictionary mapping region names to list of district names.
    regions_path : Path
        Path containing region folders.
    districts_path : Path
        Path containing district folders.
    """
    for region, districts in region_district_map.items():
        region_folder = regions_path / region

        if not region_folder.exists():
            print(f"Region folder not found: {region_folder}")
            continue

        for district in districts:
            source_district_folder = districts_path / district

            if not source_district_folder.exists():
                print(f"District folder not found: {source_district_folder}")
                continue

            destination_folder = region_folder / district

            # If already exists → remove to avoid copytree error
            if destination_folder.exists():
                shutil.rmtree(destination_folder)

            shutil.copytree(
                src=source_district_folder,
                dst=destination_folder,
            )

    print("All district folders copied successfully!")


def upload_pdf_to_s3(dir_path: Path, level: str, region: str = "eu-central-1") -> None:
    """Upload PDFs inside subfolders and their zipped folders to S3.

    Parameters
    ----------
    dir_path : Path
        Path containing subdirectories.
    level: str
        Either "region" or "district".
    region : str
        S3 region.
    """
    s3_connection = workspace.s3_connection("s3-carte-sanitaire-niger-public")

    fs = fsspec.filesystem(
        "s3",
        key=s3_connection.access_key_id,
        secret=s3_connection.secret_access_key,
        client_kwargs={"region_name": region},
    )

    for subdir in dir_path.iterdir():
        if not subdir.is_dir():
            continue

        destination_base_pdf = f"health_cov_modelling/{level}/pdf"

        pdf_files = list(subdir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF found in {subdir}")
        else:
            for pdf in pdf_files:
                s3_pdf_path = f"s3://{s3_connection.bucket_name}/{destination_base_pdf}/{pdf.name}"
                fs.put_file(str(pdf), s3_pdf_path)
                print(f"Uploaded PDF {pdf.name} to {s3_pdf_path}")


def zip_folder(dir_path: Path, output_dir: Path) -> None:
    """Zip all subdirectories contained in a parent directory.

    Parameters
    ----------
    dir_path: Path
        Path to the directory containing subdirectories to zip.
    output_dir: Path
        Directory where the .zip files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in dir_path.iterdir():
        if not subdir.is_dir():
            continue

        normalized_subdir = subdir.name.replace(" ", "_").replace("'", "")
        zip_path = output_dir / normalized_subdir

        # Remove existing zip if it exists
        if zip_path.with_suffix(".zip").exists():
            zip_path.with_suffix(".zip").unlink()

        shutil.make_archive(
            base_name=str(zip_path),
            format="zip",
            root_dir=subdir,
        )


def upload_folder_to_s3(folder_path: str, level: str, region: str = "eu-central-1") -> None:
    """Upload all zip files contained in a folder to an S3 bucket.

    Parameters
    ----------
    folder_path : Path
        Path to the directory containing zipped folders (.zip).
    level: str
        Either "region" or "district".
    region : str, optional
        S3 region. Defaults to "eu-central-1".
    """
    s3_connection = workspace.s3_connection("s3-carte-sanitaire-niger-public")

    fs = fsspec.filesystem(
        "s3",
        key=s3_connection.access_key_id,
        secret=s3_connection.secret_access_key,
        client_kwargs={"region_name": region},
    )

    for zip_folder in folder_path.iterdir():
        destination_zip = f"health_cov_modelling/{level}/zip/{zip_folder.name}"
        s3_path = f"s3://{s3_connection.bucket_name}/{destination_zip}"
        fs.put_file(zip_folder, s3_path, recursive=False)
        print(f"Uploaded {zip_folder.name} to {s3_path}")
    current_run.log_info("Résultats stockés dans le bucket s3 !")
