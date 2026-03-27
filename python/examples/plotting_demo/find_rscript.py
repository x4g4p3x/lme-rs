"""
Locate `Rscript` / `Rscript.exe`: PATH, then Windows registry + standard folders.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from typing import Optional


def _rscript_from_windows_registry() -> list[str]:
    """Paths from HKLM/HKCU SOFTWARE\\R-core\\R\\<version> InstallPath."""
    try:
        import winreg
    except ImportError:
        return []

    out: list[str] = []
    for hkey in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            key = winreg.OpenKey(hkey, r"SOFTWARE\R-core\R")
        except OSError:
            continue
        try:
            i = 0
            while True:
                try:
                    subname = winreg.EnumKey(key, i)
                except OSError:
                    break
                i += 1
                try:
                    sk = winreg.OpenKey(key, subname)
                    try:
                        install_path, _ = winreg.QueryValueEx(sk, "InstallPath")
                    finally:
                        sk.Close()
                except OSError:
                    continue
                exe = os.path.join(install_path, "bin", "Rscript.exe")
                if os.path.isfile(exe):
                    out.append(exe)
        finally:
            key.Close()
    return out


def _rscript_from_program_files_glob() -> list[str]:
    if sys.platform != "win32":
        return []
    candidates: list[str] = []
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if not base:
            continue
        pattern = os.path.join(base, "R", "R-*", "bin", "Rscript.exe")
        candidates.extend(glob.glob(pattern))
    return candidates


def find_rscript() -> Optional[str]:
    """
    Resolve Rscript for this machine.

    Order: ``shutil.which("Rscript")`` / ``which("Rscript.exe")``,
    then Windows registry (R-core InstallPath), then ``Program Files\\R\\R-*\\bin``.
    If several installs exist, pick the newest by filesystem mtime.
    """
    for name in ("Rscript", "Rscript.exe"):
        w = shutil.which(name)
        if w:
            return w

    if sys.platform != "win32":
        return None

    candidates = _rscript_from_windows_registry() + _rscript_from_program_files_glob()
    # Deduplicate, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in candidates:
        np = os.path.normpath(p)
        if np.lower() not in seen:
            seen.add(np.lower())
            uniq.append(np)

    if not uniq:
        return None
    return max(uniq, key=lambda p: os.path.getmtime(p))
