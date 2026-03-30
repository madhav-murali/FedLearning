"""
Download NIH Chest X-ray metadata or small sample
"""
import kagglehub
import os
import shutil

print("Attempting to download NIH Chest X-ray dataset metadata...")
print("Note: Full dataset is 40+ GB, looking for sample or metadata...")

try:
    # Try to download the dataset
    # This will download to Kaggle's cache directory
    print("\nDownloading NIH Chest X-ray dataset (resized version - 224x224)...")
    print("This may take a while (approximately 2 GB)...")
    
    path = kagglehub.dataset_download("khanfashee/nih-chest-x-ray-14-224x224-resized")
    
    print(f"\nDataset downloaded to: {path}")
    print("\nContents:")
    for root, dirs, files in os.walk(path):
        for file in files[:10]:  # Show first 10 files
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file} ({file_size:,} bytes)")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        break
    
    # Try to find CSV files (metadata)
    print("\nLooking for CSV metadata files...")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                csv_size = os.path.getsize(csv_path)
                print(f"\nFound CSV: {file} ({csv_size:,} bytes)")
                
                # Copy CSV to our data folder
                dest_path = os.path.join("data", f"NIH_chest_xray_{file}")
                shutil.copy(csv_path, dest_path)
                print(f"Copied to: {dest_path}")
    
    print(f"\n✅ Dataset cache location: {path}")
    print("Note: Images are cached by Kaggle. CSV metadata copied to data/ folder.")
    
except Exception as e:
    print(f"\n❌ Error downloading dataset: {e}")
    print("\nThis could be because:")
    print("1. Kaggle authentication not set up")
    print("2. Dataset is too large for automatic download")
    print("3. Network connection issues")
    
    print("\n📝 To set up Kaggle authentication:")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Create New API Token")
    print("3. Place kaggle.json in: C:\\Users\\sushmitha\\.kaggle\\")
    print("   OR set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
