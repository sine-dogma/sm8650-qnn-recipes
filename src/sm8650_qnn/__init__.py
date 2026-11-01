"""Local, resumable QAIRT recipes for SM8650. SPDX-License-Identifier: CC0-1.0."""

__version__ = "0.1.0"

import os

if os.environ.get("SM8650_OFFLINE") == "1":
    from .network import disable_network

    disable_network()
