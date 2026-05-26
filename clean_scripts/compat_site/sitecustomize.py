from __future__ import annotations

import os
import sys


def _append_site_packages() -> None:
    raw_paths = os.environ.get("DLLM_SWEEP_APPEND_SITE_PACKAGES", "")
    for raw_path in raw_paths.split(os.pathsep):
        path = raw_path.strip()
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


_append_site_packages()
