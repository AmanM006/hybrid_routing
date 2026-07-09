#!/usr/bin/env python3
"""
Verify every local Python module in main.py's import chain is listed in the
Dockerfile final-stage COPY line. Prevents v15-style packaging bugs.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STDlib_OK = True  # third-party / stdlib — not copied as .py


def _local_modules() -> set[str]:
    return {p.stem for p in ROOT.glob("*.py") if p.name != "__init__.py"}


def _parse_imports(py_path: Path) -> set[str]:
    """Return local module names imported by py_path."""
    local = _local_modules()
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in local:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                top = node.module.split(".")[0]
                if top in local:
                    found.add(top)
    return found


def _import_closure(entry: str = "main") -> set[str]:
    """Transitive closure of local imports starting from entry module."""
    local = _local_modules()
    if entry not in local:
        raise SystemExit(f"Entry module '{entry}.py' not found in {ROOT}")
    seen: set[str] = set()
    stack = [entry]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = ROOT / f"{mod}.py"
        if not path.exists():
            continue
        for dep in _parse_imports(path):
            if dep not in seen:
                stack.append(dep)
    return seen


def _dockerfile_copied_py(dockerfile: Path) -> set[str]:
    text = dockerfile.read_text(encoding="utf-8")
    copied: set[str] = set()
    in_final = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^FROM\s+(?!.*AS\s+builder)", stripped, re.I) and "builder" not in stripped.lower():
            in_final = True
        if re.match(r"^FROM\s+.*AS\s+builder", stripped, re.I):
            in_final = False
        if in_final and stripped.upper().startswith("COPY "):
            for token in stripped.split():
                if token.endswith(".py"):
                    copied.add(Path(token).name.replace(".py", ""))
    return copied


def main() -> int:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.exists():
        print("FAIL: Dockerfile not found", file=sys.stderr)
        return 1

    required = _import_closure("main")
    copied = _dockerfile_copied_py(dockerfile)

    missing = sorted(required - copied)
    extra = sorted(copied - required - {"main"})  # main is entrypoint, always copied

    print("=== Docker COPY audit (main.py import chain) ===")
    print(f"Required local modules: {sorted(required)}")
    print(f"Dockerfile COPY .py files: {sorted(copied)}")
    ok = True
    if missing:
        ok = False
        print(f"MISSING from Dockerfile COPY: {missing}", file=sys.stderr)
    else:
        print("All required modules present in Dockerfile COPY.")
    if extra:
        print(f"Note: extra .py in COPY (not in import chain): {extra}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
