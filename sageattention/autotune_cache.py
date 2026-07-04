"""
Persistent caching for Triton autotune results.
"""

import os
import pickle
from pathlib import Path
from typing import Any

# Default cache directory
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sageattention"

# Can be overridden by environment variable
_CACHE_DIR = Path(os.environ.get("SAGEATTN_CACHE_DIR", _DEFAULT_CACHE_DIR))
_CACHE_FILE = _CACHE_DIR / "autotune_cache.pkl"

# Environment variable to disable persistent caching
_PERSISTENT_ENABLED = os.environ.get("SAGEATTN_PERSISTENT_CACHE", "1").lower() not in ("0", "false", "no")


def load_cache() -> dict[Any, Any]:
    """Load autotune cache from disk."""
    if not _PERSISTENT_ENABLED:
        return {}
    
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                print(f"[SageAttention] Loaded {len(cache)} autotune configs from disk cache")
                return cache
        except Exception as e:
            print(f"[SageAttention] Warning: Failed to load cache: {e}")
    return {}


def save_cache(cache: dict[Any, Any]) -> None:
    """Save autotune cache to disk."""
    if not _PERSISTENT_ENABLED:
        return
    
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"[SageAttention] Warning: Failed to save cache: {e}")


def clear_cache() -> None:
    """Delete the persistent cache file to force re-autotuning."""
    if _CACHE_FILE.exists():
        try:
            _CACHE_FILE.unlink()
            print(f"[SageAttention] Cleared cache file: {_CACHE_FILE}")
        except Exception as e:
            print(f"[SageAttention] Warning: Failed to clear cache: {e}")
