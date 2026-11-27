"""重采样滤波核。

每个核不仅给出插值值，还通过 :meth:`ResamplingKernel.required_range`
声明计算某个输出位置时实际引用的输入样本下标区间。算子据此判断：

* 滤波器历史/前瞻是否跨越了采样率段边界；
* 一帧是否已经可以最终确认（stable）还是需要等待后续块（provisional）。

两档精度：

* :class:`LinearKernel` —— 低延迟档，两点线性插值，支撑半径 1；
* :class:`WindowedSincKernel` —— 高质量档，Kaiser 加窗 sinc，带抗混叠
  截止频率，支撑半径更大。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Protocol, Sequence


def frac_floor(x: Fraction) -> int:
    return x.numerator // x.denominator


def frac_ceil(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _sinc(x: float) -> float:
    if abs(x) < 1e-15:
        return 1.0
    px = math.pi * x
    return math.sin(px) / px


def _kaiser_i0(x: float) -> float:
    """第一类零阶修正贝塞尔函数 I0（级数实现，标准库即可）。"""

    y = x / 2.0
    term = 1.0
    total = 1.0
    for k in range(1, 64):
        term *= (y * y) / (k * k)
        total += term
        if term < 1e-16 * total:
            break
    return total


class ResamplingKernel(Protocol):
    name: str
    support_radius: int

    def required_range(self, position: Fraction) -> tuple[int, int]:
        """返回闭区间所需输入下标 ``[lo, hi]``。"""
        ...

    def interpolate(self, samples: Sequence[float], position: Fraction) -> float:
        ...


class LinearKernel:
    """两点线性插值（低延迟档）。"""

    name = "linear"
    support_radius = 1

    def required_range(self, position: Fraction) -> tuple[int, int]:
        i = frac_floor(position)
        if position.denominator == 1 or position == i:
            return i, i
        return i, i + 1

    def interpolate(self, samples: Sequence[float], position: Fraction) -> float:
        n = len(samples)
        if n == 0:
            return 0.0
        i = frac_floor(position)
        if position.denominator == 1 or position == i:
            return float(samples[min(max(i, 0), n - 1)])
        frac = float(position - i)
        y0 = float(samples[min(max(i, 0), n - 1)])
        y1 = float(samples[min(max(i + 1, 0), n - 1)])
        return y0 * (1.0 - frac) + y1 * frac


class WindowedSincKernel:
    """Kaiser 加窗 sinc 高质量核。

    ``radius`` 是单侧抽头半径，总抽头数 ``2*radius+1``。下采样时按
    ``out_rate / in_rate`` 降低截止频率以抗混叠。
    """

    name = "windowed_sinc"

    def __init__(
        self,
        in_rate: int,
        out_rate: int,
        radius: int = 8,
        beta: float = 8.6,
    ) -> None:
        if radius < 1:
            raise ValueError("sinc radius 必须 >= 1")
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self.radius = int(radius)
        self.support_radius = radius
        self.beta = float(beta)
        # 截止比例 c <= 1：h(d) = c * sinc(c*d) * kaiser(d/R)
        self.cutoff = min(1.0, out_rate / in_rate)
        self._i0_beta = _kaiser_i0(beta)

    def required_range(self, position: Fraction) -> tuple[int, int]:
        i0 = frac_floor(position)
        return i0 - self.radius, i0 + self.radius

    def _window(self, distance: float) -> float:
        if distance >= self.radius:
            return 0.0
        r = distance / self.radius
        inside = max(0.0, 1.0 - r * r)
        return _kaiser_i0(self.beta * math.sqrt(inside)) / self._i0_beta

    def interpolate(self, samples: Sequence[float], position: Fraction) -> float:
        n = len(samples)
        if n == 0:
            return 0.0
        i0 = frac_floor(position)
        x = float(position)
        acc = 0.0
        weight_sum = 0.0
        c = self.cutoff
        radius = self.radius
        for i in range(i0 - radius, i0 + radius + 1):
            d = x - i
            w = c * _sinc(c * d) * self._window(abs(d))
            idx = min(max(i, 0), n - 1)  # 边缘复制填充
            acc += w * float(samples[idx])
            weight_sum += w
        if abs(weight_sum) > 1e-12:
            return acc / weight_sum
        return float(samples[min(max(i0, 0), n - 1)])


def make_kernel(
    quality: str,
    in_rate: int,
    out_rate: int,
    filter_radius: int | None = None,
    kaiser_beta: float = 8.6,
) -> ResamplingKernel:
    """质量档位 -> 滤波核工厂。"""

    q = quality.value if hasattr(quality, "value") else str(quality)
    if q in ("high", "hq", "best", "quality"):
        radius = filter_radius if filter_radius is not None else 8
        return WindowedSincKernel(in_rate, out_rate, radius=radius, beta=kaiser_beta)
    if q in ("low_latency", "low", "fast", "realtime", "linear"):
        return LinearKernel()
    raise ValueError(f"未知 quality 档位: {quality!r}")
