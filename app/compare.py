"""双轨字幕对照：修订轨（A，左）与对照轨（B，右）的全局最优对齐。

配对规则
--------
- 仅正时长交集（``交集 > 0``，毫秒精度）可配对，零交集（含仅端点相接）不行；
- 允许 1:1，以及任一轨至多 **两条相邻 cue** 合并后与另一轨一条配对
  （2:1 / 1:2，不允许 2:2）；两条合并 cue 的有效区间取其**时间并集**，
  交集/中心偏差均按并集与对端 cue 计算；
- 未配对 cue 独立成行。

全局动态规划（避免局部贪心造成后续错位），按以下**字典序**目标最大化：

1. 交集总时长最大（毫秒；合并配对按并集区间计分）；
2. 合并次数最少；
3. 中心时间偏差总和最小（区间中点之差的绝对值之和，单位毫秒）。

DP 网格 ``dp[i][j]`` 表示对齐完左轨前 ``i`` 条、右轨前 ``j`` 条时的最优评分；
转移候选为 1:1、1:0（左独立）、0:1（右独立）、2:1、1:2，并列时按该固定顺序
选择（确定的 tie-break 偏好）。实现中三级评分被编码进单个整数（交集占据高位、
合并次数居中、偏差占据低位），以省去热循环的元组分配。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .subtitles import Cue

MAX_CUES = 5000

# 回溯标记（同时作为候选优先级，数值小者在并列时优先）
_PAIR = 0       # 1:1：(i-1,j-1)
_LEFT_ONLY = 1  # 左轨独立成行：(i-1,j)
_RIGHT_ONLY = 2  # 右轨独立成行：(i,j-1)
_PAIR_2L = 3    # 左轨两条合并配对右轨一条：(i-2,j-1)
_PAIR_2R = 4    # 左轨一条配对右轨两条合并：(i-1,j-2)


@dataclass
class AlignRow:
    """对照表中的一行：左右任一侧为空即未配对独立行。"""
    left: list[int] = field(default_factory=list)    # 修订轨 cue 序号（0 起）
    right: list[int] = field(default_factory=list)   # 对照轨 cue 序号（0 起）
    overlap_ms: int = 0      # 配对行的正时长交集（毫秒）
    center_delta_ms: int = 0  # 配对行的中心偏差（毫秒）
    merged: bool = False     # 是否含任一轨两条合并


@dataclass
class AlignResult:
    rows: list[AlignRow]
    total_overlap_ms: int
    merge_count: int
    total_center_delta_ms: int
    paired_rows: int
    left_only_rows: int
    right_only_rows: int


def _interval_ms(cue: Cue) -> tuple[int, int]:
    s = max(0, int(round(cue.start * 1000)))
    e = max(s, int(round(cue.end * 1000)))
    return s, e


def _pair_gain(la: int, lb: int, ra: int, rb: int) -> tuple[int, int]:
    """返回 (正时长交集 ms, 中心偏差 ms ×2)；无正交集返回零元（不可配对）。"""
    lo = max(la, ra)
    hi = min(lb, rb)
    if hi <= lo:
        return 0, 0
    return hi - lo, abs((la + lb) - (ra + rb))


def align_tracks(left: list[Cue], right: list[Cue]) -> AlignResult:
    """对两条轨道做全局 DP 对齐。两轨 cue 数量均不得超过 ``MAX_CUES``。

    为避免热循环里的元组分配与比较，三级字典序评分编码为单个整数：

        score = 交集和·M_RADIX·D_RADIX
                + (MERGE_OFF - 合并次数)·D_RADIX
                + (DELTA_CAP - 偏差×2总和)
    """
    if len(left) > MAX_CUES or len(right) > MAX_CUES:
        raise ValueError(f"单轨 cue 数量过多（上限 {MAX_CUES} 条）")

    n, m = len(left), len(right)
    lv = [_interval_ms(c) for c in left]
    rv = [_interval_ms(c) for c in right]

    max_end = 0
    for s, e in (*lv, *rv):
        if e > max_end:
            max_end = e
    merge_off = n + m + 2
    m_radix = 2 * merge_off
    # 单对中心偏差（×2）不超过 2·max_end；对数不超过 n+m
    delta_cap = 2 * (n + m) * (max_end + 2) + 4
    d_radix = 2 * delta_cap + 2
    ov_unit = m_radix * d_radix
    merge_penalty = d_radix

    # 相邻两条 cue 的时间并集（2:1 / 1:2 按并集计分）
    l2s = [0] + [min(lv[i - 1][0], lv[i][0]) for i in range(1, n)]
    l2e = [0] + [max(lv[i - 1][1], lv[i][1]) for i in range(1, n)]
    r2s = [0] + [min(rv[j - 1][0], rv[j][0]) for j in range(1, m)]
    r2e = [0] + [max(rv[j - 1][1], rv[j][1]) for j in range(1, m)]

    # 仅保留两行评分；回溯标记保留全网格（每格 1 字节）
    width = m + 1
    back = bytearray((n + 1) * width)
    row0 = [0] * width
    for j in range(1, m + 1):
        back[j] = _RIGHT_ONLY
    prev2 = row0  # dp[i-2]（i=1 时不会被 i>=2 的转移使用）
    prev = row0   # dp[i-1]

    for i in range(1, n + 1):
        cur = [0] * width
        back[i * width] = _LEFT_ONLY
        la, lb = lv[i - 1]
        la2, lb2 = (l2s[i - 1], l2e[i - 1]) if i >= 2 else (0, 0)
        for j in range(1, m + 1):
            ra, rb = rv[j - 1]

            best = -1
            move = _RIGHT_ONLY  # 兜底（cur[j-1] 必然可达）

            # 1:1 配对（仅正时长交集）
            lo = la if la > ra else ra
            hi = lb if lb < rb else rb
            if hi > lo:
                cand = prev[j - 1] + (hi - lo) * ov_unit \
                    - abs((la + lb) - (ra + rb))
                if cand > best:
                    best, move = cand, _PAIR

            # 左轨独立成行（评分不变）
            cand = prev[j]
            if cand > best:
                best, move = cand, _LEFT_ONLY
            # 右轨独立成行
            cand = cur[j - 1]
            if cand > best:
                best, move = cand, _RIGHT_ONLY

            # 左轨相邻两条合并配对右轨一条（交集按两左 cue 的时间并集）
            if i >= 2:
                lo = la2 if la2 > ra else ra
                hi = lb2 if lb2 < rb else rb
                if hi > lo:
                    cand = prev2[j - 1] + (hi - lo) * ov_unit \
                        - merge_penalty - abs((la2 + lb2) - (ra + rb))
                    if cand > best:
                        best, move = cand, _PAIR_2L

            # 右轨相邻两条合并配对左轨一条
            if j >= 2:
                ra2, rb2 = r2s[j - 1], r2e[j - 1]
                lo = la if la > ra2 else ra2
                hi = lb if lb < rb2 else rb2
                if hi > lo:
                    cand = cur[j - 2] + (hi - lo) * ov_unit \
                        - merge_penalty - abs((la + lb) - (ra2 + rb2))
                    if cand > best:
                        best, move = cand, _PAIR_2R

            cur[j] = best
            back[i * width + j] = move
        prev2, prev = prev, cur

    # 回溯（自 (n,m) 向 (0,0)，得到的行最后反转为时间顺序）
    rows_rev: list[AlignRow] = []
    i, j = n, m
    total_overlap = total_delta = merges = 0
    left_only = right_only = paired = 0
    while i > 0 or j > 0:
        move = back[i * width + j]
        if move == _PAIR:
            ov, dc2 = _pair_gain(lv[i - 1][0], lv[i - 1][1],
                                 rv[j - 1][0], rv[j - 1][1])
            rows_rev.append(AlignRow([i - 1], [j - 1], ov, dc2 // 2))
            total_overlap += ov
            total_delta += dc2 // 2
            paired += 1
            i -= 1
            j -= 1
        elif move == _LEFT_ONLY:
            rows_rev.append(AlignRow(left=[i - 1]))
            left_only += 1
            i -= 1
        elif move == _RIGHT_ONLY:
            rows_rev.append(AlignRow(right=[j - 1]))
            right_only += 1
            j -= 1
        elif move == _PAIR_2L:
            la2 = min(lv[i - 1][0], lv[i - 2][0])
            lb2 = max(lv[i - 1][1], lv[i - 2][1])
            ov, dc2 = _pair_gain(la2, lb2, rv[j - 1][0], rv[j - 1][1])
            rows_rev.append(AlignRow([i - 2, i - 1], [j - 1], ov,
                                     dc2 // 2, merged=True))
            total_overlap += ov
            total_delta += dc2 // 2
            paired += 1
            merges += 1
            i -= 2
            j -= 1
        else:  # _PAIR_2R
            ra2 = min(rv[j - 1][0], rv[j - 2][0])
            rb2 = max(rv[j - 1][1], rv[j - 2][1])
            ov, dc2 = _pair_gain(lv[i - 1][0], lv[i - 1][1], ra2, rb2)
            rows_rev.append(AlignRow([i - 1], [j - 2, j - 1], ov,
                                     dc2 // 2, merged=True))
            total_overlap += ov
            total_delta += dc2 // 2
            paired += 1
            merges += 1
            i -= 1
            j -= 2

    rows = list(reversed(rows_rev))
    return AlignResult(
        rows=rows,
        total_overlap_ms=total_overlap,
        merge_count=merges,
        total_center_delta_ms=total_delta,
        paired_rows=paired,
        left_only_rows=left_only,
        right_only_rows=right_only,
    )
