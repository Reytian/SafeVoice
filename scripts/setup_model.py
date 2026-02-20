#!/usr/bin/env python3
"""Download the Qwen3-ASR model for SafeVoice.

This script downloads the Qwen3-ASR-0.6B model from HuggingFace Hub.
The model is required for speech recognition.

Usage:
    python scripts/setup_model.py
    python scripts/setup_model.py --model mlx-community/Qwen3-ASR-0.6B-4bit
"""

import argparse
import sys
import time

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
AVAILABLE_MODELS = {
    "Qwen/Qwen3-ASR-0.6B": {
        "description": "0.6B parameters, fp16 (recommended, best compatibility)",
        "size": "~1.2 GB",
    },
    "mlx-community/Qwen3-ASR-0.6B-4bit": {
        "description": "0.6B parameters, 4-bit quantization (smaller, slightly less accurate)",
        "size": "~400 MB",
    },
}


def download_model(model_id: str) -> None:
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface-hub is not installed.")
        print("  pip install huggingface-hub")
        sys.exit(1)

    info = AVAILABLE_MODELS.get(model_id, {})
    size = info.get("size", "unknown size")
    desc = info.get("description", "")

    print(f"Model: {model_id}")
    if desc:
        print(f"  {desc}")
    print(f"  Estimated size: {size}")
    print()
    print("Downloading... (this may take a few minutes on first run)")
    print()

    start = time.time()
    path = snapshot_download(
        model_id,
        local_files_only=False,
    )
    elapsed = time.time() - start

    print()
    print(f"Download complete in {elapsed:.1f}s")
    print(f"Model saved to: {path}")
    print()
    print("You can now run SafeVoice:")
    print("  python run.py")


def check_model(model_id: str) -> bool:
    """Check if a model is already downloaded."""
    try:
        from huggingface_hub import try_to_load_from_cache, HfFileSystemResolvedPath
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError:
        return False

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id, local_files_only=True)
        return path is not None
    except Exception:
        return False


def list_models() -> None:
    """List available models and their download status."""
    print("Available models:")
    print()
    for model_id, info in AVAILABLE_MODELS.items():
        downloaded = check_model(model_id)
        status = "[downloaded]" if downloaded else "[not downloaded]"
        default = " (default)" if model_id == DEFAULT_MODEL else ""
        print(f"  {model_id}{default}")
        print(f"    {info['description']}")
        print(f"    Size: {info['size']}  {status}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ASR models for SafeVoice",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SafeVoice Model Setup")
    print("=" * 60)
    print()

    if args.list:
        list_models()
        return

    if check_model(args.model):
        print(f"Model '{args.model}' is already downloaded.")
        print()
        print("To force re-download, delete the cached model from:")
        print("  ~/.cache/huggingface/hub/")
        return

    download_model(args.model)


if __name__ == "__main__":
    main()
