"""
AI Image Library Sorter (OpenCLIP ViT-B/16 + WD-EVA02-Large-Tagger-v3 ONNX)
---------------------------------------------------------------------------
High-speed local batch image classifier and organizer built for 8 GB VRAM GPUs.
Categorizes images into hierarchical folders (Artwork by Franchise/Character,
Real People by Cosplay/Portrait/Group/Lifestyle, and Other by Memes/Screenshots/Nature)
while indexing all metadata and tag probabilities into a local SQLite database.
Includes an automatic dependency bootstrapper supporting both system-level Python
and automatic local './venv' creation.
"""

# =====================================================================
# 1. USER CONFIGURATION (EDIT BELOW OR LEAVE EMPTY TO BE PROMPTED)
# =====================================================================
# Path to the directory with unsorted images (e.g., r"D:\Pictures\Unsorted").
# Leave empty ("") to be prompted interactively at startup or pass via --source.
SOURCE_DIR: str = ""

# Path to the output directory where sorted folders and index.db will be created.
# Leave empty ("") to be prompted interactively at startup or pass via --output.
OUTPUT_DIR: str = ""

# File operation mode:
# False = Physically move files from SOURCE_DIR to OUTPUT_DIR
# True  = Create zero-space NTFS hardlinks (requires both folders on the same drive)
USE_HARDLINKS: bool = False

# Confidence thresholds (0.0 to 1.0):
CHARACTER_THRESHOLD: float = 0.70          # Confidence threshold for 2D anime/game characters (70%)
COPYRIGHT_THRESHOLD: float = 0.55          # Confidence threshold for 2D anime/game franchises (55%)
COSPLAY_CHAR_THRESHOLD: float = 0.42       # Relaxed character threshold for real-life 3D cosplay photos (42%)
COSPLAY_COPYRIGHT_THRESHOLD: float = 0.38  # Relaxed franchise threshold for real-life 3D cosplay photos (38%)
GENERAL_THRESHOLD: float = 0.35            # Threshold for general descriptive Danbooru tags (35%)

# Hardware & Database settings:
ONNX_VRAM_LIMIT_GB: float = 2.5            # Maximum VRAM allocated to the ONNX Runtime arena (in GB)
BATCH_COMMIT_SIZE: int = 50                # Save progress to SQLite every N processed images
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Dependency Installation Settings:
PYTORCH_INDEX_URL: str = "https://download.pytorch.org/whl/cu128"
ONNXRUNTIME_GPU_VERSION: str = "1.26.0"    # Pinned to 1.26.0 for native CUDA 12.x compatibility


# =====================================================================
# 2. DEPENDENCY CHECKER & AUTO-BOOTSTRAPPER (SYSTEM OR LOCAL ./venv)
# =====================================================================
import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import site
import sqlite3
import subprocess
import sys
import venv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Mapping of (import_module_name, pip_package_spec, distribution_name)
REQUIRED_PACKAGES: list[tuple[str, str, str]] = [
    ("torch", "torch", "torch"),
    ("torchvision", "torchvision", "torchvision"),
    ("onnxruntime", f"onnxruntime-gpu=={ONNXRUNTIME_GPU_VERSION}", "onnxruntime-gpu"),
    ("huggingface_hub", "huggingface_hub", "huggingface-hub"),
    ("PIL", "pillow", "pillow"),
    ("pandas", "pandas", "pandas"),
    ("tqdm", "tqdm", "tqdm"),
    ("open_clip", "open_clip_torch", "open-clip-torch"),
    ("numpy", "numpy", "numpy"),
]


def get_missing_packages() -> list[tuple[str, str, str]]:
    """
    Checks whether all required packages are installed in the currently running
    Python environment. Also verifies that 'onnxruntime-gpu' is pinned to 1.26.0
    so CUDA 12.x DLLs from PyTorch load properly.
    """
    missing: list[tuple[str, str, str]] = []
    for module_name, pip_spec, dist_name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, pip_spec, dist_name))
            continue

        # Specifically verify that onnxruntime-gpu==1.26.0 is installed (not CPU-only or 1.27+ CUDA 13)
        if dist_name == "onnxruntime-gpu":
            try:
                installed_ver = importlib.metadata.version("onnxruntime-gpu")
                if installed_ver != ONNXRUNTIME_GPU_VERSION:
                    missing.append((module_name, pip_spec, dist_name))
            except importlib.metadata.PackageNotFoundError:
                missing.append((module_name, pip_spec, dist_name))

    return missing


def get_local_venv_python(venv_dir: Path) -> Path:
    """Returns the path to the Python executable inside the given virtual environment directory."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def is_running_in_venv(venv_dir: Path) -> bool:
    """Checks if the script is currently executing inside the target local './venv'."""
    try:
        return Path(sys.prefix).resolve() == venv_dir.resolve()
    except OSError:
        return False


def install_packages_with_pip(python_exe: Path, packages_to_install: list[tuple[str, str, str]]) -> None:
    """
    Installs the specified packages using the target Python interpreter's pip.
    Installs torch and torchvision from the CUDA 12.8 wheel index first, followed by remaining packages.
    """
    print(f"\n[Bootstrapper] Upgrading pip using: {python_exe}")
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])

    pip_specs = [spec for _, spec, _ in packages_to_install]
    torch_specs = [spec for spec in pip_specs if spec in ("torch", "torchvision")]
    other_specs = [spec for spec in pip_specs if spec not in ("torch", "torchvision")]

    if torch_specs:
        print(f"\n[Bootstrapper] Installing PyTorch with CUDA support ({', '.join(torch_specs)})...")
        subprocess.check_call([
            str(python_exe),
            "-m",
            "pip",
            "install",
            *torch_specs,
            "--index-url",
            PYTORCH_INDEX_URL,
        ])

    if other_specs:
        print(f"\n[Bootstrapper] Installing dependencies ({', '.join(other_specs)})...")
        subprocess.check_call([
            str(python_exe),
            "-m",
            "pip",
            "install",
            *other_specs,
        ])


def ensure_dependencies() -> None:
    """
    1. Checks if all required packages are already available in the current (system) Python.
       If yes, continues execution immediately without requiring a venv.
    2. If packages are missing, checks if a local './venv' already exists in the project directory.
       If './venv' exists and has all packages, transparently re-launches inside './venv'.
    3. Otherwise, prompts the user with a single-key choice:
       [1] Create './venv' in the project directory, install packages there, and run from './venv'
       [2] Install missing packages directly into the current system Python environment
       [Q] Quit
    """
    missing = get_missing_packages()
    if not missing:
        return

    project_dir = Path(__file__).resolve().parent
    local_venv_dir = project_dir / "venv"
    venv_python = get_local_venv_python(local_venv_dir)

    # If the user already created './venv' on a previous run and launched the script via global python,
    # automatically switch to './venv' without bothering them.
    if venv_python.exists() and not is_running_in_venv(local_venv_dir):
        print(f"[Bootstrapper] Found existing local environment at '{local_venv_dir}'. Switching interpreter...")
        result = subprocess.run([str(venv_python), *sys.argv])
        sys.exit(result.returncode)

    missing_names = ", ".join(spec for _, spec, _ in missing)
    print("\n" + "=" * 75)
    print("MISSING OR INCOMPATIBLE PYTHON PACKAGES DETECTED:")
    print(f"  {missing_names}")
    print("=" * 75)
    print("How would you like to set up the required packages?")
    print(f"  [1] Create a local virtual environment in '{local_venv_dir}' and run from there (Recommended)")
    print(f"  [2] Install packages globally into the current system Python ({sys.executable})")
    print("  [Q] Quit without installing")
    print("=" * 75)

    while True:
        try:
            choice = input("Select an option [1/2/Q] (default: 1): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user.")
            sys.exit(0)

        if choice in ("", "1"):
            if not venv_python.exists():
                print(f"\n[Bootstrapper] Creating virtual environment in '{local_venv_dir}'...")
                venv.create(local_venv_dir, with_pip=True)

            # Install all required packages into the newly created ./venv
            install_packages_with_pip(venv_python, REQUIRED_PACKAGES)
            print("\n[Bootstrapper] Setup complete! Launching script inside './venv'...\n")
            result = subprocess.run([str(venv_python), *sys.argv])
            sys.exit(result.returncode)

        if choice == "2":
            install_packages_with_pip(Path(sys.executable), missing)
            print("\n[Bootstrapper] System-level installation complete! Restarting script...\n")
            result = subprocess.run([sys.executable, *sys.argv])
            sys.exit(result.returncode)

        if choice == "q":
            print("Exiting without installing packages.")
            sys.exit(0)

        print("Invalid selection. Please enter 1, 2, or Q.")


# Run dependency check before importing third-party libraries
ensure_dependencies()


# =====================================================================
# 3. AUTOMATIC CUDA 12.x / cuDNN DLL REGISTRATION (SYSTEM & VENV)
# =====================================================================
def register_cuda_dlls() -> None:
    """
    Locates CUDA and cuDNN shared libraries across both virtual environments
    and system/user site-packages (e.g., torch/lib and nvidia/*/bin) and registers
    them in PATH so onnxruntime-gpu==1.26.0 can initialize CUDAExecutionProvider.
    """
    candidate_roots: list[Path] = [Path(sys.prefix) / "Lib" / "site-packages"]

    # Include global system site-packages and user-level site-packages
    for sp in site.getsitepackages():
        candidate_roots.append(Path(sp))
    user_sp = site.getusersitepackages()
    if user_sp:
        candidate_roots.append(Path(user_sp))

    dll_dirs: list[Path] = []
    seen_roots: set[Path] = set()

    for root in candidate_roots:
        try:
            resolved_root = root.resolve()
        except OSError:
            continue
        if resolved_root in seen_roots or not resolved_root.exists():
            continue
        seen_roots.add(resolved_root)

        nvidia_root = resolved_root / "nvidia"
        if nvidia_root.exists():
            dll_dirs.extend([p for p in nvidia_root.glob("*/bin") if p.is_dir()])

        torch_lib = resolved_root / "torch" / "lib"
        if torch_lib.exists():
            dll_dirs.append(torch_lib)

    for directory in dll_dirs:
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass


register_cuda_dlls()

import numpy as np
import onnxruntime as ort
import open_clip
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile
from tqdm import tqdm

# Preload DLLs if supported by the installed ONNX Runtime build
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls()
    except Exception:
        pass

# Allow loading slightly corrupted or truncated images downloaded from social media
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =====================================================================
# 4. PATH RESOLUTION, HELPER FUNCTIONS & DATABASE INITIALIZATION
# =====================================================================
def resolve_directory_path(
    cli_value: str | None,
    config_value: str,
    prompt_message: str,
    must_exist: bool = False,
) -> Path:
    """
    Resolves a directory path from CLI arguments, top-of-file configuration constants,
    or interactive user input if neither is provided. Strips surrounding quotes so users
    can drag-and-drop folders directly into the terminal window.
    """
    raw_path = (cli_value or config_value or "").strip().strip('"').strip("'")

    while True:
        if not raw_path:
            try:
                raw_path = input(prompt_message).strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled by user.")
                sys.exit(0)

        if not raw_path:
            print("Error: Path cannot be empty. Please enter a valid directory path.")
            continue

        candidate = Path(raw_path).expanduser().resolve()
        if must_exist and not candidate.exists():
            print(f"Error: Directory does not exist: {candidate}")
            raw_path = ""
            continue
        if must_exist and not candidate.is_dir():
            print(f"Error: Path is not a directory: {candidate}")
            raw_path = ""
            continue

        return candidate


def sanitize_windows_name(name: str) -> str:
    r"""Replaces characters forbidden in Windows folder/file names (: / \ * ? " < > |) with underscores."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip().rstrip(". ")
    return cleaned or "unknown"


def split_char_and_anime(best_char: str | None, best_copyright: str | None) -> tuple[str | None, str | None]:
    """
    Extracts (franchise_name, character_name).
    If the character tag embeds the franchise in parentheses, e.g., 'mona_(genshin_impact)',
    it splits them cleanly; otherwise it pairs the character with the detected copyright tag.
    """
    anime_name = best_copyright
    char_name = best_char
    if best_char:
        match = re.match(r"^(.+?)_\((.+)\)$", best_char)
        if match:
            char_name = match.group(1)
            if not anime_name:
                anime_name = match.group(2)
    return anime_name, char_name


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initializes the SQLite database used to store image metadata for instant re-querying."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            filepath TEXT PRIMARY KEY,
            original_path TEXT,
            filename TEXT,
            macro_category TEXT,
            sub_category TEXT,
            subfolder TEXT,
            anime_title TEXT,
            character_name TEXT,
            character_conf REAL,
            clip_conf REAL,
            rating TEXT,
            top_tags TEXT
        )
    """)
    conn.commit()
    return conn


# =====================================================================
# 5. VISION MODELS: WD-EVA02-LARGE-TAGGER-V3 (ONNX) & OPENCLIP (FP16)
# =====================================================================
class AnimeTaggerONNX:
    """
    Runs SmilingWolf/wd-eva02-large-tagger-v3 via ONNX Runtime on GPU.
    Extracts content rating, general tags, copyright (franchise), and character tags.
    """

    def __init__(self) -> None:
        repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        print("[1/2] Loading WD-EVA02-Large-Tagger-v3 (ONNX)...")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")

        df = pd.read_csv(csv_path)
        self.tag_names = df["name"].tolist()
        self.rating_idx = df.index[df["category"] == 9].tolist()     # Category 9: Rating
        self.general_idx = df.index[df["category"] == 0].tolist()    # Category 0: General tags
        self.copyright_idx = df.index[df["category"] == 3].tolist()  # Category 3: Franchise / Anime title
        self.character_idx = df.index[df["category"] == 4].tolist()  # Category 4: Character name

        # Restrict ONNX arena memory growth so it coexists peacefully on 8 GB VRAM GPUs
        cuda_options = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": int(ONNX_VRAM_LIMIT_GB * 1024 * 1024 * 1024),
            "cudnn_conv_algo_search": "HEURISTIC",
        }
        providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        active_providers = self.session.get_providers()
        print(f"--> Active ONNX Execution Provider: {active_providers[0]}")

        self.input_name = self.session.get_inputs()[0].name
        self.target_size = 448

        # Meta-tags in category 3 that do not represent an actual media franchise
        self.ignored_copyrights = {
            "original",
            "commission",
            "request",
            "fanbox_reward",
            "patreon_reward",
            "skeb_commission",
        }

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Pads the image to a square with a white background, resizes to 448x448, and converts RGB to BGR."""
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        canvas.alpha_composite(img)
        img = canvas.convert("RGB")

        max_dim = max(img.size)
        pad_left = (max_dim - img.size[0]) // 2
        pad_top = (max_dim - img.size[1]) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, (pad_left, pad_top))

        if max_dim != self.target_size:
            padded = padded.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)

        arr = np.asarray(padded, dtype=np.float32)
        return arr[:, :, ::-1]  # Convert RGB -> BGR expected by WDv3

    def predict(self, img: Image.Image) -> dict:
        """Runs inference and returns both standard (2D) and relaxed (3D cosplay) character/franchise matches."""
        inp = np.expand_dims(self.preprocess(img), axis=0)
        probs = self.session.run(None, {self.input_name: inp})[0][0]

        # 1. Content Rating (general, sensitive, questionable, explicit)
        ratings = {self.tag_names[i]: float(probs[i]) for i in self.rating_idx}
        best_rating = max(ratings.items(), key=lambda x: x[1])[0] if ratings else "general"

        # 2. Top-20 General Tags
        gen_tags = {
            self.tag_names[i]: round(float(probs[i]), 3)
            for i in self.general_idx
            if probs[i] > GENERAL_THRESHOLD
        }
        gen_tags = dict(sorted(gen_tags.items(), key=lambda x: x[1], reverse=True)[:20])

        # 3. Franchise (Copyright) - Standard & Cosplay thresholds
        copy_candidates = {
            self.tag_names[i]: round(float(probs[i]), 3)
            for i in self.copyright_idx
            if probs[i] >= COSPLAY_COPYRIGHT_THRESHOLD and self.tag_names[i] not in self.ignored_copyrights
        }
        best_copy_any = max(copy_candidates.items(), key=lambda x: x[1]) if copy_candidates else (None, 0.0)
        best_copyright = best_copy_any[0] if best_copy_any[1] >= COPYRIGHT_THRESHOLD else None
        cosplay_copyright = best_copy_any[0]

        # 4. Character Name - Standard & Cosplay thresholds
        char_candidates = {
            self.tag_names[i]: round(float(probs[i]), 3)
            for i in self.character_idx
            if probs[i] >= COSPLAY_CHAR_THRESHOLD
        }
        best_char_any = max(char_candidates.items(), key=lambda x: x[1]) if char_candidates else (None, 0.0)
        best_char = best_char_any[0] if best_char_any[1] >= CHARACTER_THRESHOLD else None
        best_char_conf = best_char_any[1] if best_char else 0.0

        cosplay_char = best_char_any[0]
        cosplay_char_conf = best_char_any[1]

        return {
            "gen_tags": gen_tags,
            "rating": best_rating,
            "best_copyright": best_copyright,
            "best_char": best_char,
            "char_conf": best_char_conf,
            "cosplay_copyright": cosplay_copyright,
            "cosplay_char": cosplay_char,
            "cosplay_char_conf": cosplay_char_conf,
        }


class HierarchicalClassifierCLIP:
    """
    Uses OpenCLIP ViT-B-16 in FP16 (~350 MB VRAM) to perform both:
    1. Macro-classification (Real_People vs Artwork vs Other)
    2. Zero-cost sub-classification inside Real_People and Other using the same image embedding.
    """

    def __init__(self) -> None:
        print("[2/2] Loading OpenCLIP ViT-B-16 (FP16)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device=self.device
        )
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

        # Stage 1: Macro Category Prompts
        self.macro_prompts = [
            "a real life photo of a person, human portrait, selfie, real people, real human face, cosplay photo",
            "an anime illustration, 2d digital art, anime character, manga drawing, fanart, cel shading",
            "a meme with text, screenshot of a phone or chat, software interface, text document, comic strip with text bubbles",
            "a real photo of nature, landscape, animal, cat, dog, car, food, interior, object without people",
        ]
        self.macro_categories = ["Real_People", "Artwork", "Other", "Other"]

        # Stage 2A: Sub-categories for Real_People
        self.real_sub_prompts = [
            "a cosplay photo of a real person wearing an anime costume, wig, character outfit at a convention or studio",
            "a close-up selfie, face portrait, headshot, mirror selfie of a real person",
            "a real person in a swimsuit, bikini, lingerie, beachwear, glamour model",
            "a group photo of multiple real people, friends, crowd, party, two or more people together",
            "a full body street photo, everyday casual outfit, fashion or lifestyle photo of a real person",
        ]
        self.real_sub_categories = [
            "Cosplay",
            "Selfies_and_Portraits",
            "Swimsuit_and_Glamour",
            "Group_Photos",
            "Full_Body_and_Lifestyle",
        ]

        # Stage 2B: Sub-categories for Other
        self.other_sub_prompts = [
            "an internet meme with text captions, funny image macro, demotivator, reaction meme",
            "a screenshot of a computer screen, smartphone app, chat messages, social media post, document",
            "a photo of a cute animal, cat, dog, pet, wildlife, or natural landscape, mountains, forest, sea",
            "a photo of food, car, vehicle, architecture, room interior, electronics, or everyday object",
        ]
        self.other_sub_categories = [
            "Memes_and_Comics",
            "Screenshots_and_Text",
            "Nature_and_Animals",
            "Objects_and_Vehicles",
        ]

        with torch.no_grad():
            self.macro_features = self._encode_prompts(self.macro_prompts)
            self.real_sub_features = self._encode_prompts(self.real_sub_prompts)
            self.other_sub_features = self._encode_prompts(self.other_sub_prompts)

    def _encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    def classify(self, img: Image.Image) -> dict:
        """Encodes the image once and computes probabilities across macro and sub-category heads."""
        with torch.no_grad():
            inp = self.preprocess(img).unsqueeze(0).to(self.device)
            if self.device == "cuda":
                inp = inp.half()
            img_feats = self.model.encode_image(inp)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            # 1. Macro category
            macro_probs = (100.0 * img_feats @ self.macro_features.T).softmax(dim=-1)[0]
            macro_idx = int(macro_probs.argmax().item())
            macro_cat = self.macro_categories[macro_idx]
            macro_conf = round(float(macro_probs[macro_idx].item()), 3)

            # 2. Real_People sub-category
            real_probs = (100.0 * img_feats @ self.real_sub_features.T).softmax(dim=-1)[0]
            real_idx = int(real_probs.argmax().item())
            real_sub = self.real_sub_categories[real_idx]

            # 3. Other sub-category
            other_probs = (100.0 * img_feats @ self.other_sub_features.T).softmax(dim=-1)[0]
            other_idx = int(other_probs.argmax().item())
            other_sub = self.other_sub_categories[other_idx]

            return {
                "macro_cat": macro_cat,
                "macro_conf": macro_conf,
                "real_sub": real_sub,
                "other_sub": other_sub,
            }


# =====================================================================
# 6. ROUTING & FILE PLACEMENT LOGIC
# =====================================================================
def determine_routing(clip_res: dict, wd_res: dict) -> tuple[str, str, str, str | None, str | None, float]:
    """
    Combines OpenCLIP and WD-EVA02-Tagger predictions to decide:
    (macro_category, sub_category, relative_subfolder, anime_title, character_name, character_conf)
    """
    gen_tags = wd_res["gen_tags"]
    rating = wd_res["rating"]
    macro_cat = clip_res["macro_cat"]

    is_real_photo = "real_life" in gen_tags or "photo_(medium)" in gen_tags
    is_cosplay_tag = "cosplay" in gen_tags

    # Rule 1: Memes and screenshots without a dominant 2D character go to Other
    if ("meme" in gen_tags or "screenshot" in gen_tags) and not wd_res["best_char"]:
        macro_cat = "Other"

    # Rule 2: If WD-Tagger sees a strong 2D character/franchise and NO real-life photo tags, it's Artwork
    if (wd_res["best_char"] or wd_res["best_copyright"]) and not is_real_photo and not is_cosplay_tag:
        if clip_res["real_sub"] != "Cosplay":
            macro_cat = "Artwork"

    # Rule 3: If WD-Tagger explicitly sees 'real_life' with human tags, keep in Real_People
    if is_real_photo and (
        "1girl" in gen_tags or "1boy" in gen_tags or "multiple_girls" in gen_tags or "multiple_boys" in gen_tags
    ):
        if macro_cat == "Artwork":
            macro_cat = "Real_People"

    # --- ROUTING FOR "Artwork" ---
    if macro_cat == "Artwork":
        anime_title, clean_char = split_char_and_anime(wd_res["best_char"], wd_res["best_copyright"])
        char_conf = wd_res["char_conf"]

        if anime_title or clean_char:
            safe_anime = sanitize_windows_name(anime_title) if anime_title else "_Unknown_Franchise"
            if clean_char:
                safe_char = sanitize_windows_name(clean_char)
                sub_cat = f"{safe_anime}/{safe_char}"
                subfolder = f"Artwork\\{safe_anime}\\{safe_char}"
            else:
                sub_cat = f"{safe_anime}/_No_Character"
                subfolder = f"Artwork\\{safe_anime}\\_No_Character"
        else:
            sub_cat = "_Original_or_Unknown"
            subfolder = "Artwork\\_Original_or_Unknown"

        return macro_cat, sub_cat, subfolder, anime_title, clean_char, char_conf

    # --- ROUTING FOR "Real_People" ---
    if macro_cat == "Real_People":
        real_sub = clip_res["real_sub"]

        # Refine sub-category using reliable WD-Tagger signals
        if is_cosplay_tag or wd_res["cosplay_char"] or wd_res["cosplay_copyright"]:
            real_sub = "Cosplay"
        elif (
            rating in ("questionable", "explicit")
            or "swimsuit" in gen_tags
            or "bikini" in gen_tags
            or "underwear" in gen_tags
        ):
            real_sub = "Swimsuit_and_Glamour"
        elif (
            "multiple_girls" in gen_tags
            or "multiple_boys" in gen_tags
            or "2girls" in gen_tags
            or "2boys" in gen_tags
        ):
            real_sub = "Group_Photos"
        elif "selfie" in gen_tags or "portrait" in gen_tags or "close-up" in gen_tags:
            real_sub = "Selfies_and_Portraits"

        if real_sub == "Cosplay":
            anime_title, clean_char = split_char_and_anime(wd_res["cosplay_char"], wd_res["cosplay_copyright"])
            char_conf = wd_res["cosplay_char_conf"]
            if anime_title or clean_char:
                safe_anime = sanitize_windows_name(anime_title) if anime_title else "_Unknown_Franchise"
                if clean_char:
                    safe_char = sanitize_windows_name(clean_char)
                    subfolder = f"Real_People\\Cosplay\\{safe_anime}\\{safe_char}"
                else:
                    subfolder = f"Real_People\\Cosplay\\{safe_anime}\\_No_Character"
            else:
                subfolder = "Real_People\\Cosplay\\_Unidentified_Cosplay"
            return macro_cat, "Cosplay", subfolder, anime_title, clean_char, char_conf

        subfolder = f"Real_People\\{real_sub}"
        return macro_cat, real_sub, subfolder, None, None, 0.0

    # --- ROUTING FOR "Other" ---
    other_sub = clip_res["other_sub"]
    if "meme" in gen_tags or "comic" in gen_tags:
        other_sub = "Memes_and_Comics"
    elif "screenshot" in gen_tags:
        other_sub = "Screenshots_and_Text"

    subfolder = f"Other\\{other_sub}"
    return "Other", other_sub, subfolder, None, None, 0.0


def place_file(src: Path, dst_folder: Path, use_hardlinks: bool) -> Path:
    """Moves (or hardlinks) the file to the destination folder, resolving filename collisions safely."""
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst = dst_folder / src.name
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return dst
        except OSError:
            pass
        dst = dst_folder / f"{src.stem}_{abs(hash(str(src))) % 100000}{src.suffix}"

    if use_hardlinks:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))
    return dst


def print_summary(conn: sqlite3.Connection, db_path: Path) -> None:
    """Prints a formatted summary table of the SQLite index database."""
    print("\n" + "=" * 75)
    print(f"DATABASE SUMMARY ({db_path})")
    print("=" * 75)

    df_cat = pd.read_sql_query(
        """
        SELECT macro_category AS 'Macro Category',
               CASE WHEN macro_category = 'Artwork' AND character_name IS NOT NULL
                    THEN 'Identified_Character'
                    ELSE sub_category END AS 'Sub-Category',
               COUNT(*) AS 'Count'
        FROM images
        GROUP BY macro_category,
                 CASE WHEN macro_category = 'Artwork' AND character_name IS NOT NULL
                      THEN 'Identified_Character'
                      ELSE sub_category END
        ORDER BY macro_category, Count DESC
        """,
        conn,
    )
    if not df_cat.empty:
        print(df_cat.to_string(index=False))

    print("\nTOP-15 RECOGNIZED [FRANCHISE / CHARACTER] PAIRS:")
    df_char = pd.read_sql_query(
        """
        SELECT macro_category AS 'Type',
               COALESCE(anime_title, '_Unknown_Franchise') AS 'Franchise / Anime',
               character_name AS 'Character',
               COUNT(*) AS 'Images',
               ROUND(AVG(character_conf) * 100, 1) || '%' AS 'Avg Confidence'
        FROM images
        WHERE character_name IS NOT NULL
        GROUP BY macro_category, anime_title, character_name
        ORDER BY COUNT(*) DESC
        LIMIT 15
        """,
        conn,
    )
    if df_char.empty:
        print("No characters recognized above threshold yet.")
    else:
        print(df_char.to_string(index=False))
    print("=" * 75)


# =====================================================================
# 7. CLI ENTRY POINT
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-speed local AI image sorter using OpenCLIP (FP16) and WD-EVA02-Large-Tagger-v3 (ONNX)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the unsorted source directory (overrides SOURCE_DIR in script)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the sorted output directory (overrides OUTPUT_DIR in script)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 = all)")
    parser.add_argument(
        "--hardlinks",
        action="store_true",
        default=USE_HARDLINKS,
        help="Create NTFS hardlinks instead of moving files",
    )
    parser.add_argument("--summary-only", action="store_true", help="Print SQLite database statistics and exit")
    args = parser.parse_args()

    # Resolve output directory first (required for both sorting and --summary-only)
    output_dir = resolve_directory_path(
        cli_value=args.output,
        config_value=OUTPUT_DIR,
        prompt_message="Enter path to the OUTPUT directory (for sorted folders & index.db): ",
        must_exist=False,
    )
    db_path = output_dir / "index.db"

    conn = init_db(db_path)
    if args.summary_only:
        print_summary(conn, db_path)
        conn.close()
        return

    # Resolve source directory (must exist on disk)
    source_dir = resolve_directory_path(
        cli_value=args.source,
        config_value=SOURCE_DIR,
        prompt_message="Enter path to the SOURCE directory (unsorted images): ",
        must_exist=True,
    )

    cursor = conn.cursor()
    print(f"Scanning source directory: {source_dir} ...")

    # Avoid scanning inside output_dir or local './venv' if located inside source_dir
    resolved_output = output_dir.resolve()
    local_venv_dir = (Path(__file__).resolve().parent / "venv").resolve()
    all_files = [
        p for p in source_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.resolve() != resolved_output
        and resolved_output not in p.resolve().parents
        and local_venv_dir not in p.resolve().parents
    ]
    print(f"Found images in source directory: {len(all_files)}")

    if args.limit > 0:
        all_files = all_files[: args.limit]
        print(f"Test mode enabled: processing first {len(all_files)} files.")

    if not all_files:
        print_summary(conn, db_path)
        conn.close()
        return

    clip_cls = HierarchicalClassifierCLIP()
    wd_tagger = AnimeTaggerONNX()
    batch_records = []

    for filepath in tqdm(all_files, desc="Sorting & moving images"):
        try:
            # Load image into memory and immediately close the file handle so Windows allows shutil.move()
            with Image.open(filepath) as raw_img:
                img = raw_img.convert("RGBA")

            clip_res = clip_cls.classify(img)
            wd_res = wd_tagger.predict(img)

            macro_cat, sub_cat, subfolder, anime_title, clean_char, char_conf = determine_routing(clip_res, wd_res)

            target_folder = output_dir / subfolder
            new_filepath = place_file(filepath, target_folder, use_hardlinks=args.hardlinks)

            batch_records.append((
                str(new_filepath),
                str(filepath),
                new_filepath.name,
                macro_cat,
                sub_cat,
                subfolder,
                anime_title,
                clean_char,
                char_conf,
                clip_res["macro_conf"],
                wd_res["rating"],
                json.dumps(wd_res["gen_tags"], ensure_ascii=False),
            ))

            if len(batch_records) >= BATCH_COMMIT_SIZE:
                cursor.executemany(
                    "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch_records,
                )
                conn.commit()
                batch_records.clear()

        except Exception as e:
            print(f"\n[Skipped] Error processing {filepath.name}: {e}")

    if batch_records:
        cursor.executemany(
            "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch_records,
        )
        conn.commit()

    print_summary(conn, db_path)
    conn.close()


if __name__ == "__main__":
    main()