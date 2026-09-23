"""双轨字幕时间轴全局对齐（动态规划）。

把「修订轨」A 与上传的「对照轨」B 按时间交集配对，允许的对齐形式：

- 一对一：``A[i] ↔ B[j]``
- 二对一：``A[i] + A[i+1]``（相邻两条合并）↔ ``B[j]``
- 一对二：``A[i] ↔ B[j] + B[j+1]``
- 未配对：某一轨的单条 cue 独立成行

配对（含合并组）要求**正时长交集**；同一轨两条相邻 cue 合并后按其时间
并集 ``[min start, max end]`` 与另一轨计交集（并集中的间隙也算覆盖）。

优化目标按字典序依次为：

1. 全部配对的交集总时长最大；
2. 合并（二对一 / 一对二）次数最少；
3. 配对两侧时间中心偏差总和最小。

DP 覆盖全部跨轨配对可能，避免局部贪心造成后续错位。实现上只保留最近
两行 DP 数值，回溯指针用 bytearray 网格存放，可承受数千条 cue 的字幕。
"""
from __future__ import annotations

from dataclasses import dataclass

from .subtitles import Cue

_EPS = 1e-9

# 回溯指针编码
_SKIP_A = 1     # A[i] 独立成行
_SKIP_B = 2     # B[j] 独立成行
_PAIR_11 = 3    # A[i] ↔ B[j]
_PAIR_A2 = 4    # A[i]+A[i+1] ↔ B[j]
_PAIR_B2 = 5    # A[i] ↔ B[j]+B[j+1]


@dataclass
class AlignRow:
    """对照表中的一行。a/b 为对应轨 cue 的下标（0 起），单边为空表示未配对。"""
    kind: str               # "pair" | "a_only" | "b_only"
    a: list[int]
    b: list[int]
    overlap: float = 0.0    # 配对交集时长（秒），未配对行为 0
    center_delta: float | None = None  # 配对中心偏差（秒）


def _overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    lo = s1 if s1 > s2 else s2
    hi = e1 if e1 < e2 else e2
    return hi - lo if hi > lo else 0.0


def _span(cues: list[Cue], idxs: list[int]) -> tuple[float, float]:
    """一组 cue（含相邻两条合并）的时间并集。"""
    start = min(cues[k].start for k in idxs)
    end = max(cues[k].end for k in idxs)
    return start, end


def _center(start: float, end: float) -> float:
    return (start + end) / 2.0


def _better(cand: tuple[float, int, float],
            best: tuple[float, int, float]) -> bool:
    """DP 取值 (交集总时长, -合并次数, -中心偏差总和)，字典序取大。"""
    return cand > best


def align(a: list[Cue], b: list[Cue]) -> list[AlignRow]:
    """对两条时间轴做全局对齐，返回按时间顺序排列的对照行。"""
    n, m = len(a), len(b)
    width = m + 1
    pointers = bytearray((n + 1) * width)

    # 边界行 i == n：A 已耗尽，剩余 B 只能逐条独立成行
    nxt = [(0.0, 0, 0.0)] * width
    for j in range(m - 1, -1, -1):
        pointers[n * width + j] = _SKIP_B

    nxt2: list[tuple[float, int, float]] | None = None  # i+2 行
    for i in range(n - 1, -1, -1):
        cur: list[tuple[float, int, float] | None] = [None] * width
        cur[m] = nxt[m]
        pointers[i * width + m] = _SKIP_A
        ai, ai1 = a[i], a[i + 1] if i + 1 < n else None

        for j in range(m - 1, -1, -1):
            bj = b[j]
            # 候选 1：A[i] 不配对
            best, best_code = nxt[j], _SKIP_A
            # 候选 2：B[j] 不配对
            cand_val = cur[j + 1]
            if _better(cand_val, best):
                best, best_code = cand_val, _SKIP_B

            # 候选 3：一对一
            ov = _overlap(ai.start, ai.end, bj.start, bj.end)
            if ov > _EPS:
                base = nxt[j + 1]
                dev = abs(_center(ai.start, ai.end)
                          - _center(bj.start, bj.end))
                cand_val = (base[0] + ov, base[1], base[2] - dev)
                if _better(cand_val, best):
                    best, best_code = cand_val, _PAIR_11

            # 候选 4：二对一（A 相邻两条合并，交集按 A 的时间并集计）
            if ai1 is not None and nxt2 is not None:
                us = ai.start if ai.start < ai1.start else ai1.start
                ue = ai.end if ai.end > ai1.end else ai1.end
                ov = _overlap(us, ue, bj.start, bj.end)
                if ov > _EPS:
                    base = nxt2[j + 1]
                    dev = abs(_center(us, ue) - _center(bj.start, bj.end))
                    cand_val = (base[0] + ov, base[1] - 1, base[2] - dev)
                    if _better(cand_val, best):
                        best, best_code = cand_val, _PAIR_A2

            # 候选 5：一对二（B 相邻两条合并，交集按 B 的时间并集计）
            if j + 1 < m:
                bj1 = b[j + 1]
                us = bj.start if bj.start < bj1.start else bj1.start
                ue = bj.end if bj.end > bj1.end else bj1.end
                ov = _overlap(ai.start, ai.end, us, ue)
                if ov > _EPS:
                    base = nxt[j + 2]
                    dev = abs(_center(ai.start, ai.end) - _center(us, ue))
                    cand_val = (base[0] + ov, base[1] - 1, base[2] - dev)
                    if _better(cand_val, best):
                        best, best_code = cand_val, _PAIR_B2

            cur[j] = best
            pointers[i * width + j] = best_code

        nxt2, nxt = nxt, cur  # type: ignore[assignment]

    # 回溯，逐行生成结果（指针从 (0,0) 起按时间顺序给出）
    rows: list[AlignRow] = []
    i = j = 0
    while i < n or j < m:
        code = pointers[i * width + j]
        if code == _SKIP_A:
            rows.append(AlignRow("a_only", [i], []))
            i += 1
        elif code == _SKIP_B:
            rows.append(AlignRow("b_only", [], [j]))
            j += 1
        else:
            ia = [i]
            ib = [j]
            if code == _PAIR_A2:
                ia.append(i + 1)
                i += 2
                j += 1
            elif code == _PAIR_B2:
                ib.append(j + 1)
                i += 1
                j += 2
            else:  # _PAIR_11
                i += 1
                j += 1
            sa, ea = _span(a, ia)
            sb, eb = _span(b, ib)
            ov = max(0.0, min(ea, eb) - max(sa, sb))
            dev = abs(_center(sa, ea) - _center(sb, eb))
            rows.append(AlignRow(
                "pair", ia, ib, round(ov, 3), round(dev, 3)))
    return rows
