"""Opt-in guard loaded by worker interpreters. SPDX-License-Identifier: CC0-1.0."""
import os

if os.environ.get("SM8650_OFFLINE") == "1":
    from sm8650_qnn.network import disable_network

    disable_network()
