import requests
import zipfile
import os
from pathlib import Path
import sys
import os
import time
from requests.exceptions import ChunkedEncodingError

# Add src to system path
sys.path.append(str(Path.cwd().parent))

# import modules
from utils.utils import measure_execution_time

@measure_execution_time
def download_shp(url: str, filename: str, unzip: bool = False, deleteZip: bool = True, chunk_size: int = 8192) -> None:
    """
    Downloads a file from a URL and optionally extracts it if it's a ZIP file.

    Parameters:
        url (str): The URL of the file to download.
        filename (str): The local filename to save the downloaded file as.
        unzip (bool): Whether to extract the file if it's a ZIP file. Default is False.
        chunk_size (int): The size of chunks for streaming downloads. Default is 8192 bytes.
        deleteZip (bool): Whether to delete zipped files after extraction.
    """
    try:
        filename_print = os.path.basename(filename)
        print(f'Downloading file {filename_print}...')

        # Ensure the target directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Download the file with streaming for large files
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {filename_print}")

        # Check if the file exists and is non-empty before unzipping
        if unzip and os.path.exists(filename) and os.path.getsize(filename) > 0:
            file_base_name = os.path.splitext(os.path.basename(filename))[0]
            extract_directory = Path(filename).parent / file_base_name
            try:
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(extract_directory)
                print(f"Extracted files to: {extract_directory}")
                
                # Delete the ZIP file after extraction if desired
                if deleteZip:
                    Path(filename).unlink(missing_ok=True)
                    if not Path(filename).exists():
                        print(f"Deleted ZIP file: {filename}")
                    else:
                        print(f"Failed to delete ZIP file: {filename}")
                    
            except zipfile.BadZipFile:
                print(f"Error: {filename} is not a valid ZIP file.")
    except requests.RequestException as e:
        print(f"Download failed: {e}")
    except OSError as e:
        print(f"File operation failed: {e}")

@measure_execution_time
def download_GeoTIFF(url: str, filename: str, chunk_size: int = 8192) -> None:
    """
    Downloads a file from a URL.

    Parameters:
        url (str): The URL of the file to download.
        filename (str): The local filename to save the downloaded file as.
        chunk_size (int): The size of chunks for streaming downloads. Default is 8192 bytes.
    """
    try:
        filename_print = os.path.basename(filename)
        print(f'Downloading file {filename_print}...')

        # Ensure the target directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Download the file with streaming for large files
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the total size of the file from the server
        total_size = int(response.headers.get('Content-Range', '').split('/')[-1]) if 'Content-Range' in response.headers else int(response.headers.get('Content-Length', 0))
        downloaded_size = 0

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Display progress
                    print(f"Downloaded {downloaded_size}/{total_size} bytes ({(downloaded_size / total_size) * 100:.2f}%)", end="\r")

        print(f"Downloaded: {filename_print}")


    except requests.RequestException as e:
        print(f"Download failed: {e}")
    except OSError as e:
        print(f"File operation failed: {e}")

@measure_execution_time
def download_large_file(url: str, destination: str, max_retries: int =3, chunk_size: int = 1024 * 1024):
    """
    Downloads a file from a given URL in chunks, with support for resuming the download if interrupted and retrying on errors.

    Parameters:
    -----------
        url (str): The URL of the file to download.
        destination (str): The local file path where the downloaded file will be saved.
        max_retries (int): The maximum number of times to retry the download on error. Default is 3.
        chunk_size (int): The size of chunks to download the file in bytes. Default is 1 MB.

    Functionality:
    --------------
        - Checks if a partially downloaded file exists at the destination.
        - Resumes the download from where it left off using the 'Range' HTTP header.
        - Downloads the file in chunks (default: 10 MB) to minimize memory usage.
        - Displays download progress in the console.
        - Implements retry logic for handling connection interruptions such as ChunkedEncodingError.
        - Ensures compatibility with servers supporting HTTP range requests.

    Note:
    -----
        - Ensure the server supports partial downloads (HTTP status code 206).
        - If the server does not support range requests, the download will restart from the beginning.
        - The function will retry up to 3 times on certain connection errors before failing.

    Raises:
    -------
        requests.exceptions.RequestException: If there is an issue with the HTTP request beyond retry attempts.
    """


    retries = 0

    while retries < max_retries:
        try:
            # Get the file size of the partially downloaded file if it exists
            downloaded_size = 0
            if os.path.exists(destination):
                downloaded_size = os.path.getsize(destination)

            # Get the total size of the file from the server
            headers = {"Range": f"bytes={downloaded_size}-"}  # Resume from the downloaded size

            with requests.get(url, headers=headers, stream=True) as response:
                # Ensure the server supports partial downloads
                if response.status_code == 206 or response.status_code == 200:
                    total_size = int(response.headers.get('Content-Range', '').split('/')[-1]) if 'Content-Range' in response.headers else int(response.headers.get('Content-Length', 0))

                    # Open the file in append mode and write chunks
                    with open(destination, "ab") as file:
                        print(f"Starting download: {destination} ({downloaded_size}/{total_size} bytes)")

                        for chunk in response.iter_content(chunk_size=chunk_size): 
                            if chunk:  # Filter out keep-alive chunks
                                file.write(chunk)
                                downloaded_size += len(chunk)

                                # Display progress
                                print(f"Downloaded {downloaded_size}/{total_size} bytes ({(downloaded_size / total_size) * 100:.2f}%)", end="\r")

                    print(f"\nDownload completed: {destination}")
                    return
                else:
                    print(f"Failed to download file. Server responded with status code {response.status_code}.")
                    return
        except ChunkedEncodingError as e:
            retries += 1
            print(f"ChunkedEncodingError occurred: {e}. Retrying {retries}/{max_retries}...")
            time.sleep(2**retries)

    raise Exception("Failed to download file after multiple retries.")





