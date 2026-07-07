import os
import argparse
from huggingface_hub import snapshot_download

def download_oaizib(output_dir="data/oaizib"):
    """
    Downloads the OAIZIB-CM dataset from Hugging Face.
    This contains MRI volumes and corresponding cartilage/bone masks.
    """
    print(f"Downloading OAIZIB-CM to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # We use snapshot_download to pull the dataset.
    # To avoid downloading the entire massive dataset during a POC, 
    # we can restrict it to a few files if we know the structure, 
    # or just pull the whole thing if required.
    # For now, we will pull the repository.
    try:
        snapshot_download(
            repo_id="YongchengYAO/OAIZIB-CM",
            repo_type="dataset",
            local_dir=output_dir,
            max_workers=4
        )
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download OAIZIB: {e}")

def download_tcia_samples(output_dir="data/tcia_samples"):
    """
    Downloads a few sample DICOM cases from TCIA for the Week 1 requirement.
    Uses tcia-utils.
    """
    print(f"Downloading TCIA sample cases to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        from tcia_utils import nbia
        # The Osteoarthritis Initiative (OAI) collection on TCIA
        # We will just pull a small sample of series for testing.
        # This gets a list of series in the OAI collection
        series_data = nbia.getSeries(collection="OAI")
        
        if series_data and len(series_data) > 0:
            # Take the first 5 series for the POC
            sample_series = [s['SeriesInstanceUID'] for s in series_data[:5]]
            print(f"Found {len(sample_series)} series to download.")
            
            for uid in sample_series:
                print(f"Downloading Series: {uid}")
                nbia.downloadSeries(SeriesInstanceUID=uid, path=output_dir)
            print("TCIA download complete.")
        else:
            print("No series found for OAI collection on TCIA.")
    except ImportError:
        print("tcia-utils not installed. Run `pip install tcia-utils`.")
    except Exception as e:
        print(f"Failed to download from TCIA: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Knee Datasets for POC")
    parser.add_argument("--source", choices=["huggingface", "tcia", "all"], default="all",
                        help="Which data source to pull from.")
    args = parser.parse_args()

    if args.source in ["huggingface", "all"]:
        download_oaizib()
    
    if args.source in ["tcia", "all"]:
        download_tcia_samples()
