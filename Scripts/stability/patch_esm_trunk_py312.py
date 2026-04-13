#!/usr/bin/env python3
"""
Python 3.12 对 dataclass 可变默认值更严格；fair-esm 2.x 的 ESMFold 需打两处补丁：

  - esm/esmfold/v1/trunk.py     StructureModuleConfig
  - esm/esmfold/v1/esmfold.py   ESMFoldConfig.trunk / FoldingTrunkConfig

用法（装过 fair-esm 后执行；可重复执行，已修补的会跳过）:
  python patch_esm_trunk_py312.py
"""
from __future__ import annotations

import os
import re
import sys


def _ensure_dataclass_field_import(text: str) -> str:
    old = "from dataclasses import dataclass\n"
    new = "from dataclasses import dataclass, field\n"
    if "from dataclasses import dataclass, field" in text:
        return text
    if old in text:
        return text.replace(old, new, 1)
    return text


def patch_trunk(path: str) -> str:
    """返回 'skip' | 'ok' | 'fail'"""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if "field(default_factory=StructureModuleConfig)" in text:
        return "skip"

    pattern = r"^(\s*)structure_module: StructureModuleConfig = StructureModuleConfig\(\)\s*$"
    repl = r"\1structure_module: StructureModuleConfig = field(default_factory=StructureModuleConfig)"
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n != 1:
        return "fail"

    new_text = _ensure_dataclass_field_import(new_text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return "ok"


def patch_esmfold(path: str) -> str:
    """修补 ESMFoldConfig.trunk = FoldingTrunkConfig()"""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if "field(default_factory=FoldingTrunkConfig)" in text:
        return "skip"

    # trunk: T.Any = FoldingTrunkConfig()
    pattern = r"^(\s*)trunk:\s*T\.Any\s*=\s*FoldingTrunkConfig\(\)\s*$"
    repl = r"\1trunk: T.Any = field(default_factory=FoldingTrunkConfig)"
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n != 1:
        return "fail"

    new_text = _ensure_dataclass_field_import(new_text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return "ok"


def main() -> int:
    try:
        import esm
    except ImportError:
        print("请先 pip install fair-esm", file=sys.stderr)
        return 1

    root = os.path.dirname(esm.__file__)
    trunk_path = os.path.join(root, "esmfold", "v1", "trunk.py")
    esmfold_path = os.path.join(root, "esmfold", "v1", "esmfold.py")

    for p in (trunk_path, esmfold_path):
        if not os.path.isfile(p):
            print(f"找不到: {p}", file=sys.stderr)
            return 1

    r = patch_trunk(trunk_path)
    if r == "ok":
        print(f"已修补: {trunk_path}")
    elif r == "skip":
        print(f"已打过补丁，跳过: {trunk_path}")
    else:
        print(
            "trunk.py 未找到 structure_module = StructureModuleConfig()，请核对 fair-esm 版本。",
            file=sys.stderr,
        )
        return 1

    r2 = patch_esmfold(esmfold_path)
    if r2 == "ok":
        print(f"已修补: {esmfold_path}")
    elif r2 == "skip":
        print(f"已打过补丁，跳过: {esmfold_path}")
    else:
        print(
            "esmfold.py 未找到 trunk = FoldingTrunkConfig()，请手动改为:\n"
            "  trunk: T.Any = field(default_factory=FoldingTrunkConfig)\n"
            "并确保 from dataclasses import dataclass, field",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
