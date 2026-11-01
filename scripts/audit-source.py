#!/usr/bin/env python3
"""Check the publishable source inventory. SPDX-License-Identifier: CC0-1.0."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "scripts", "tests", "models", "targets", "recipes", "requirements", "docs", ".github"}
ROOT_FILES = {"README.md", "LICENSE", "NOTICE", "CONTRIBUTING.md", "pyproject.toml", ".gitignore", ".gitattributes"}
FORBIDDEN_SUFFIXES = {".onnx", ".data", ".dlc", ".bin", ".raw", ".npz", ".safetensors", ".dll", ".so", ".exe", ".zip", ".gz", ".log"}
PATTERNS = {
    "credential-like value": re.compile(r"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Windows user/workspace path": re.compile(r"[A-Za-z]:[\\/](?:Users|WORKSPACE)[\\/][A-Za-z0-9]", re.I),
    "Linux user/workspace path": re.compile(r"/(?:home/[A-Za-z0-9_.-]+/|mnt/[a-z]/(?:Users|WORKSPACE)/[A-Za-z0-9])", re.I),
}


def files():
    result = subprocess.run(["git", "ls-files", "-z", "--cached"], cwd=ROOT, capture_output=True)
    if result.returncode == 0 and result.stdout:
        return [Path(name) for name in result.stdout.decode("utf-8").split("\0") if name]
    found = [Path(name) for name in ROOT_FILES if (ROOT / name).is_file()]
    for directory in DIRECTORIES:
        if (ROOT / directory).is_dir():
            found += [p.relative_to(ROOT) for p in (ROOT / directory).rglob("*")
                      if p.is_file() and not any(part == "__pycache__" or part.endswith(".egg-info") for part in p.parts)]
    return found


def main():
    failures = []
    total = 0
    paths = files()
    for relative in sorted(paths):
        path = ROOT / relative
        if relative.parts[0] not in DIRECTORIES and relative.as_posix() not in ROOT_FILES:
            failures.append(f"{relative.as_posix()}: unexpected publishable location")
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            failures.append(f"{relative.as_posix()}: source symlink or escaped path")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith(".env"):
            failures.append(f"{relative.as_posix()}: generated asset or local configuration")
            continue
        if path.stat().st_size > 1024*1024:
            failures.append(f"{relative.as_posix()}: unexpectedly large source file")
            continue
        content = path.read_text(encoding="utf-8")
        total += path.stat().st_size
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                failures.append(f"{relative.as_posix()}: {label}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                local = target.split("#", 1)[0]
                if local and not (path.parent / local).is_file():
                    failures.append(f"{relative.as_posix()}: broken relative documentation link")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(json.dumps({"source_files": len(paths), "source_bytes": total, "audit": "PASSED"}, indent=2))


if __name__ == "__main__":
    main()
