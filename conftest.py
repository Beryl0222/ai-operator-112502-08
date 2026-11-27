"""Pytest 启动钩子：让 src/ 布局在未执行 pip install -e . 时也可导入。

unittest 路径由 README 中的 ``pip install -e .`` 保证；此文件只改善
``pytest`` 直接运行时的健壮性，不属于算子运行时代码。
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
