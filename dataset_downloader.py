from pathlib import Path
import kagglehub

# Specify target directory relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
custom_path = str(PROJECT_ROOT / "datasets")

# Download latest version directly to your custom path
path = kagglehub.dataset_download(
    "thoughtvector/customer-support-on-twitter", 
    output_dir=custom_path
)

print("Path to dataset files:", path)
