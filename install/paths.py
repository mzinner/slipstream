"""Immutable program files and writable per-user data, for source or release."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
DATA = Path.home() / "Library/Application Support/Slipstream-v2" if PACKAGED else ROOT
MODELS = DATA / "models" if PACKAGED else ROOT / "install/models"
RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
_V2_BIN = ROOT / ("engine/slipstream-v2" if PACKAGED else "build/slipstream-v2")
_V1_BIN = ROOT / ("engine/slipstream" if PACKAGED else "build/slipstream")
_SPLASH_BIN = ROOT / ("engine/splash" if PACKAGED else "build/splash")
BINARY = (
    _V2_BIN if _V2_BIN.exists()
    else _V1_BIN if _V1_BIN.exists()
    else _SPLASH_BIN if _SPLASH_BIN.exists()
    else _V2_BIN
)
