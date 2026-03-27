import kagglehub
import os

def download_data():
    # Download the latest version of the NIH dataset used in this benchmark
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    print("Path to dataset files:", path)
    return path

if __name__ == "__main__":
    dataset_path = download_data()
    # Optionally, list files
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            print(os.path.join(root, file))
