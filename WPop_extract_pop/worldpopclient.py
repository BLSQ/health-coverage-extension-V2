from pathlib import Path

import requests


class WorldPopClient:
    """Mini client for the WorldPop REST API.

    Source: https://data.worldpop.org/GIS/Population
    """

    def __init__(self, url: str = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A"):
        """Initialize the client with a specific project alias and subproject.

        NOTE: We only point to population data /Population.

        Parameters
        ----------
        url : str
            The base URL for the WorldPop data download.
        """
        self.base_url = url


    def download_data_for_country(
        self,
        output_dir: Path,
        year: int,
    ) -> Path:
        """Download and save the WorldPop raster dataset for a given country and year.

        This operation is atomic. A partial download will not result in a corrupt
        final file.

        Parameters
        ----------
        output_dir : Path
            Directory to save the GeoTIFF file.
        year : str
            Year to filter the dataset (e.g., "2025").

        Returns
        -------
        Path
            Full path to the saved GeoTIFF file.

        Raises
        ------
        IOError
            If the file download or disk write fails.
        """

        try:
            base_url, fname = self._build_url(year)
            destination_path = output_dir / "population_raw.tif"
        except Exception as e:
            raise ValueError(
                f"Could not determine download details for Niger - {year}: {e}"
            ) from e

        self._atomic_download(f"{base_url}{fname}", destination_path)
        return destination_path


    def _build_url(self, year: int) -> tuple[str, str]:
        """
        Build download URL.

        Parameter
        ----------
        year : str, optional
            Year of interest (2015--2030).
        Returns
        -------
        url : str
            Download URL.
        """
        return (f"{self.base_url}/{year}/NER/v1/100m/constrained/", self.target_tif_filename(year))
    

    def target_tif_filename(self, year: str) -> str:
        """Get the expected target filename for Niger and a given year.

        Parameter
        ----------
        year : str, optional
            Year of interest (2015--2030).
        Returns
        -------
        str
            Expected filename of the downloaded GeoTIFF file.
        """
        return f"ner_pop_{year}_CN_100m_R2025A_v1.tif"


    @staticmethod
    def _atomic_download(
        url: str, destination_path: Path, session: requests.Session | None = None
    ) -> None:
        """
        Downloads a file from a URL to a destination path atomically.

        It downloads to a temporary file first and renames it upon success,
        preventing partial/corrupt files.

        Parameters
        ----------
        url : str
            The URL of the file to download.
        destination_path : Path
            The final path to save the file.
        session : requests.Session, optional
            An existing requests session to use for the download (reusing connections).

        Raises
        ------
        requests.HTTPError
            If the download fails with a non-200 status code.
        OSError
            If the file cannot be written to disk.
        """
        # Download to a temporary path
        temp_path = destination_path.with_suffix(destination_path.suffix + ".part")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        http_client = session or requests

        try:
            with http_client.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                with Path.open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                        f.write(chunk)
            # If download is successful, rename the temp file to the final destination
            temp_path.rename(destination_path)

        except (requests.RequestException, OSError) as e:
            raise OSError(f"Failed to download or write file from {url}: {e}") from e
        finally:
            if temp_path.exists():  # Clean up the partial file
                try:
                    temp_path.unlink()
                except OSError as e:
                    raise OSError(f"Error removing partial file {temp_path}: {e}") from e
