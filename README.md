# Local AI Image Sorter (OpenCLIP + WD-EVA02-Large-Tagger-v3)

A high-speed, 100% local batch image classification and organization pipeline designed for consumer GPUs with **8 GB VRAM** (such as the NVIDIA RTX 3070 Laptop).

Built to organize tens of thousands of unsorted images (e.g., 45,000+ social media downloads) into clean, hierarchical directories while indexing all detected tags, character probabilities, franchises, and content ratings into a local **SQLite database** (`index.db`) for instant re-sorting or AI agent integration.

---

## Key Features

* **Zero-Hassle Dependency Bootstrapper (No Manual `venv` Activation Needed):**
  * Checks if required packages are installed in your **system Python** first—if present, runs immediately.
  * If packages are missing, automatically detects any existing `./venv` in the project directory and switches to it seamlessly.
  * On first run (if dependencies are missing), offers a **one-key interactive prompt**: press `[1]` to auto-create `./venv` and install everything locally, or press `[2]` to install packages globally at the system level.
* **Dual-Model Vision Architecture (~2.5 GB VRAM footprint):**
  * **OpenCLIP (`ViT-B-16` in FP16):** Performs zero-shot macro-classification (`Artwork`, `Real_People`, `Other`) and fine-grained sub-categorization (`Selfies_and_Portraits`, `Group_Photos`, `Swimsuit_and_Glamour`, `Full_Body_and_Lifestyle`, `Memes_and_Comics`, `Screenshots_and_Text`, `Nature_and_Animals`, `Objects_and_Vehicles`) in a single forward pass.
  * **WD-EVA02-Large-Tagger-v3 (`ONNX Runtime GPU`):** Recognizes **15,000+ anime/game characters**, **copyright franchises** (anime/game titles), content ratings (`general`, `sensitive`, `questionable`, `explicit`), and general visual tags with >90% accuracy.
* **Smart 3D Cosplay Detection:** Automatically lowers character and franchise confidence thresholds when a real-life cosplay photo is detected, organizing cosplayers into `Real_People/Cosplay/<Franchise>/<Character>`.
* **Flexible Configuration & Interactive Prompts:** Configure paths and thresholds directly at the top of `sorter.py`, pass them via CLI flags (`--source`, `--output`), or simply run the script and enter (or drag-and-drop) folder paths interactively at startup.
* **SQLite Metadata Indexing:** Saves all predictions into `index.db` every 50 images. Supports safe interruption (`Ctrl+C`) and instant SQL-based re-organization without re-running GPU inference.
* **Safe File Operations:** Supports both physical file moving (`shutil.move`) and zero-overhead NTFS hardlinks (`--hardlinks`).

---

## Output Directory Structure

```text
Output_Directory/
├── index.db                                  # SQLite database with all tags, ratings & probabilities
├── Artwork/
│   ├── neon_genesis_evangelion/
│   │   ├── ayanami_rei/
│   │   └── _No_Character/
│   ├── genshin_impact/
│   │   └── mona/
│   ├── vocaloid/
│   │   └── hatsune_miku/
│   └── _Original_or_Unknown/
├── Real_People/
│   ├── Cosplay/
│   │   ├── nier_automata/
│   │   │   └── 2b/
│   │   └── _Unidentified_Cosplay/
│   ├── Selfies_and_Portraits/
│   ├── Swimsuit_and_Glamour/
│   ├── Group_Photos/
│   └── Full_Body_and_Lifestyle/
└── Other/
    ├── Memes_and_Comics/
    ├── Screenshots_and_Text/
    ├── Nature_and_Animals/
    └── Objects_and_Vehicles/
```

---

## Hardware & Software Requirements

* **OS:** Windows 10 / 11 (64-bit) or Linux
* **GPU:** NVIDIA GPU with at least 4–8 GB VRAM (tested on RTX 3070 Laptop 8 GB)
* **Python:** Python 3.11 – 3.13 (64-bit)

---

## Quick Start (Automatic One-Key Setup)

You do **not** need to manually create or activate a virtual environment before running `sorter.py`.

### 1. Clone the Repository & Run the Script

```powershell
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>

python sorter.py
```

### 2. Automatic Dependency Check & One-Key Installer

When you run `python sorter.py`:
1. **If all packages are already installed at the system level:** The script starts immediately.
2. **If `./venv` was created on a previous run:** Even if you run global `python sorter.py` without activating `./venv`, the script automatically detects `./venv` and switches to it behind the scenes.
3. **If packages are missing on first launch:** The script displays an interactive prompt:

```text
===========================================================================
MISSING OR INCOMPATIBLE PYTHON PACKAGES DETECTED:
  torch, torchvision, onnxruntime-gpu==1.26.0, huggingface_hub, pillow, pandas, tqdm, open_clip_torch, numpy
===========================================================================
How would you like to set up the required packages?
  [1] Create a local virtual environment in './venv' and run from there (Recommended)
  [2] Install packages globally into the current system Python
  [Q] Quit without installing
===========================================================================
Select an option [1/2/Q] (default: 1):
```

* Press **`1` (or `Enter`)** to automatically build `./venv` in the project folder, install PyTorch (`cu128`) and `onnxruntime-gpu==1.26.0`, and immediately continue execution.
* Press **`2`** to install all dependencies directly into your system-wide Python installation so no `venv` folder is needed at all.

---

## Manual Installation (Optional)

If you prefer to install dependencies manually via `pip` (either globally or in a custom environment):

```powershell
# 1. Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

# 2. Install ONNX Runtime GPU (pinned to 1.26.0) and vision libraries
pip install onnxruntime-gpu==1.26.0 huggingface_hub pillow pandas tqdm open_clip_torch numpy
```

> **Why is `onnxruntime-gpu` pinned to `1.26.0`?**
> Starting with `onnxruntime-gpu==1.27.0`, default PyPI wheels are compiled against **CUDA 13.0** and require `cublasLt64_13.dll`. Because PyTorch (`cu128` / `cu126`) bundles **CUDA 12.x** libraries, installing `onnxruntime-gpu>=1.27.0` fails to load `CUDAExecutionProvider` and falls back to slow CPU execution.
> Pinning **`onnxruntime-gpu==1.26.0`** (which uses CUDA 12.x) ensures native compatibility with PyTorch's bundled CUDA 12.x and cuDNN 9 DLLs out of the box. Furthermore, `sorter.py` automatically scans both virtual environment and system `site-packages` directories (`torch/lib` and `nvidia/*/bin`) and registers their DLL paths at startup.

---

## Configuration

At the very top of `sorter.py` (Section `1. USER CONFIGURATION`), you can customize the default paths and classification thresholds:

```python
SOURCE_DIR: str = ""                 # Leave empty ("") to be prompted at startup, or set e.g. r"D:\Unsorted"
OUTPUT_DIR: str = ""                 # Leave empty ("") to be prompted at startup, or set e.g. r"D:\Sorted"
USE_HARDLINKS: bool = False          # False = move files; True = create NTFS hardlinks
CHARACTER_THRESHOLD: float = 0.70    # Confidence threshold for 2D anime characters (70%)
COPYRIGHT_THRESHOLD: float = 0.55    # Confidence threshold for 2D anime/game franchises (55%)
COSPLAY_CHAR_THRESHOLD: float = 0.42 # Relaxed threshold for 3D real-life cosplay characters (42%)
GENERAL_THRESHOLD: float = 0.35      # Threshold for general Danbooru tags (35%)
```

If `SOURCE_DIR` or `OUTPUT_DIR` are left as empty strings (`""`) and not passed via command-line arguments, the script will interactively prompt you to enter (or drag-and-drop) the directory paths when launched.

---

## Usage

### 1. Interactive Launch (Prompted Paths)

Simply run `python sorter.py` from any terminal—it will automatically use your system packages (or local `./venv`) and ask you for the output and source directories:

```powershell
python sorter.py
```

### 2. Test Run on the First 100 Images (Using CLI Flags)

Before running on tens of thousands of files, verify that `CUDAExecutionProvider` is active and inspect the output folder structure on a small batch:

```powershell
python sorter.py --source "D:\Unsorted_Images" --output "D:\Sorted_Images" --limit 100
```

Check the console output at startup to confirm GPU acceleration:
```text
[1/2] Loading WD-EVA02-Large-Tagger-v3 (ONNX)...
--> Active ONNX Execution Provider: CUDAExecutionProvider
```

### 3. Full Batch Run (Move Mode)

To process and move all remaining images from the source folder into the sorted hierarchy:

```powershell
python sorter.py --source "D:\Unsorted_Images" --output "D:\Sorted_Images"
```

### 4. Non-Destructive Mode (NTFS Hardlinks)

If your source and output directories are on the same drive and you want to keep original files untouched while consuming **0 bytes** of extra disk space, add `--hardlinks` (or set `USE_HARDLINKS = True` in `sorter.py`):

```powershell
python sorter.py --source "D:\Unsorted_Images" --output "D:\Sorted_Images" --hardlinks
```

### 5. View Database Summary Only

To inspect category counts and top recognized characters without running inference:

```powershell
python sorter.py --output "D:\Sorted_Images" --summary-only
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `SOURCE_DIR` (or interactive prompt) | Path to the folder containing unsorted images |
| `--output` | `OUTPUT_DIR` (or interactive prompt) | Path where sorted folders and `index.db` will be created |
| `--limit` | `0` (All files) | Process only the first `N` images for testing |
| `--hardlinks` | `USE_HARDLINKS` (`False`) | Create NTFS hardlinks instead of moving source files |
| `--summary-only` | `False` | Print SQLite statistics and exit immediately |

---

## SQLite Database Schema (`index.db`)

Every processed image is recorded in the `images` table inside `<output>/index.db`:

| Column | Type | Description |
|---|---|---|
| `filepath` | `TEXT PRIMARY KEY` | Final path of the sorted image |
| `original_path` | `TEXT` | Original path before sorting |
| `filename` | `TEXT` | Image filename |
| `macro_category` | `TEXT` | `Artwork`, `Real_People`, or `Other` |
| `sub_category` | `TEXT` | Sub-category (e.g., `Selfies_and_Portraits`, `Cosplay`, `Memes_and_Comics`) |
| `subfolder` | `TEXT` | Relative subfolder path inside the output directory |
| `anime_title` | `TEXT` | Detected franchise / copyright tag (nullable) |
| `character_name` | `TEXT` | Detected character name (nullable) |
| `character_conf` | `REAL` | Confidence score (`0.0` – `1.0`) for character detection |
| `clip_conf` | `REAL` | Confidence score (`0.0` – `1.0`) for OpenCLIP macro classification |
| `rating` | `TEXT` | Content rating: `general`, `sensitive`, `questionable`, or `explicit` |
| `top_tags` | `TEXT` | JSON dictionary of top-20 general Danbooru tags and their probabilities |