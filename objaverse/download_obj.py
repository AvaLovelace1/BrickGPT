
import objaverse
import pandas as pd
import os
import multiprocessing
import shutil

def download_objaverse_assets(csv_path: str, output_dir: str = "objaverse_assets"):
    """
    Downloads Objaverse assets specified in a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing UIDs.
        output_dir: Directory to save the downloaded assets.
    """
    
    # Read the CSV to get the UIDs
    df = pd.read_csv(csv_path)
    uids = df["UID"].tolist()
    print(f"Found {len(uids)} UIDs to download.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    """
    Use multiprocessing to download in parallel (handled by objaverse library)
    objaverse.load_objects will download the objects to the ~/.objaverse cache 
    and return a map of {uid: file_path}
    """
    objects = objaverse.load_objects(
        uids=uids,
        download_processes=multiprocessing.cpu_count()
    )
    
    print("\nDownload complete!")
    print("-" * 30)
    
    for uid, path in objects.items():
        # Getting file name and extension:
        filename = os.path.basename(path)
        extension = os.path.splitext(path)[1]
        dest_path = os.path.join(output_dir, f"{uid}{extension}")
        
        # Copying file to local directory for easier access
        shutil.copy2(path, dest_path)
        print(f"Saved: {dest_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(SCRIPT_DIR, "objaversepp_small.csv")
    OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIR, "assets")
    
    download_objaverse_assets(CSV_FILE, OUTPUT_DIRECTORY)
