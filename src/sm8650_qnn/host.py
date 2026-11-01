"""SDK host paths. SPDX-License-Identifier: CC0-1.0."""
from dataclasses import dataclass
import os
from pathlib import Path
import platform


@dataclass(frozen=True)
class Host:
    sdk: Path
    system: str

    @classmethod
    def current(cls, sdk):
        system = platform.system()
        if system not in {"Linux", "Windows"} or platform.machine().lower() not in {
            "x86_64", "amd64"
        }:
            raise RuntimeError("These recipes require an x86-64 Linux or Windows host")
        return cls(Path(sdk).expanduser().resolve(), system)

    @property
    def tag(self):
        return "x86_64-windows-msvc" if self.system == "Windows" else "x86_64-linux-clang"

    @property
    def bin(self):
        return self.sdk / "bin" / self.tag

    @property
    def lib(self):
        return self.sdk / "lib" / self.tag

    def executable(self, name):
        return self.bin / (name + (".exe" if self.system == "Windows" else ""))

    def library(self, name):
        return self.lib / (name + ".dll" if self.system == "Windows" else "lib" + name + ".so")

    def environment(self, sdk_python=False):
        """Keep SDK Python packages separate from the export environment."""
        env = os.environ.copy()
        source = Path(__file__).resolve().parents[1]
        guard = Path(__file__).resolve().parent / "guard"
        paths = [str(guard), str(source)]
        if sdk_python:
            paths.append(str(self.sdk / "lib" / "python"))
        env.update(
            QAIRT_SDK_ROOT=str(self.sdk), QNN_SDK_ROOT=str(self.sdk),
            SM8650_OFFLINE="1", PYTHONPATH=os.pathsep.join(paths),
            PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
            TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
            GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never",
        )
        env["PATH"] = os.pathsep.join([str(self.lib), str(self.bin), env.get("PATH", "")])
        if self.system == "Linux":
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [str(self.lib), env.get("LD_LIBRARY_PATH", "")]
            ).rstrip(os.pathsep)
        return env
