import os
from zipfile import ZipFile

# Define paths
raw_data_dir = './data/raw'
processed_data_dir = './data/processed'
zip_file_path = os.path.join(raw_data_dir, 'dataset.zip')

# Ensure raw and processed data directories exist
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# Kaggle dataset details
# Replace 'username/dataset-name' with the actual Kaggle dataset identifier
kaggle_dataset = 'username/dataset-name'

# Download Kaggle dataset
def download_kaggle_dataset(dataset, save_dir):
    print("Downloading dataset from Kaggle...")
    os.system(f'kaggle datasets download -d {dataset} -p {save_dir}')
    print("Download complete!")

# Extract zip file to processed directory
def unzip_data(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

if __name__ == "__main__":
    # Step 1: Download the dataset from Kaggle to raw data folder
    download_kaggle_dataset(kaggle_dataset, raw_data_dir)

    # Step 2: Find the downloaded zip file (assuming only one zip file is downloaded)
    zip_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.zip')]
    if zip_files:
        zip_file_path = os.path.join(raw_data_dir, zip_files[0])

        # Step 3: Unzip the downloaded file to the processed data folder
        unzip_data(zip_file_path, processed_data_dir)

    print("Dataset download and extraction to processed folder complete!")
