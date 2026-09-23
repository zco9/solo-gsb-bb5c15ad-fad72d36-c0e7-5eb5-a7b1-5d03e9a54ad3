"""双轨对齐纯逻辑校验（仅需标准库）：

    python3 tests/aligner_check.py

验证全局动态规划：正时长交集配对、合并组按时间并集计分、
「交集总时长最大 → 合并次数最少 → 中心偏差最小」字典序目标，
以及局部贪心会错位时仍能得到全局最优。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.aligner import align  # noqa: E402
from app.subtitles import Cue  # noqa: E402


def check(label, cond):
    if not cond:
        raise AssertionError(label)
    print(f"ok - {label}")


def kinds(rows):
    return [r.kind for r in rows]


def pairing(rows):
    return [(r.a, r.b, r.kind) for r in rows]


# 完全一致的时间轴：逐对一对一
same_a = [Cue(0, 2, "a1"), Cue(2, 4, "a2"), Cue(4, 6, "a3")]
same_b = [Cue(0, 2, "b1"), Cue(2, 4, "b2"), Cue(4, 6, "b3")]
rows = align(same_a, same_b)
check("相同时间轴全部一对一",
      kinds(rows) == ["pair"] * 3
      and all(len(r.a) == len(r.b) == 1 for r in rows))
check("交集等于 cue 时长且中心偏差为 0",
      [r.overlap for r in rows] == [2.0, 2.0, 2.0]
      and all(r.center_delta == 0.0 for r in rows))

# B 轨把 A 的前两条合并：期望 A1+A2 ↔ B1
merged_b = [Cue(0, 4, "b1"), Cue(4, 6, "b2")]
rows = align(same_a, merged_b)
check("一对二合并（B 两条并一条）", pairing(rows) == [
    ([0, 1], [0], "pair"), ([2], [1], "pair"),
])
check("合并组交集按并集计为 4s", rows[0].overlap == 4.0)

# A 轨一条对 B 两条
merged_a = [Cue(0, 4, "a1"), Cue(4, 6, "a2")]
rows = align(merged_a, same_b)
check("二对一合并（A 一条对 B 两条）", pairing(rows) == [
    ([0], [0, 1], "pair"), ([1], [2], "pair"),
])

# 局部贪心会错位：
# A1=[0,3) A2=[3,10)；B1=[2,5) 与 A1 交集 1、与 A2 交集 2；
# B2=[5,10) 与 A2 交集 5。
# 贪心顺序匹配会给出 (A1,B1)+(A2,B2) 总交集 6；
# 全局最优是 A1 独立 + A2 与 B1+B2 合并配对，交集 0+7=7。
greedy_a = [Cue(0, 3, "a1"), Cue(3, 10, "a2")]
greedy_b = [Cue(2, 5, "b1"), Cue(5, 10, "b2")]
rows = align(greedy_a, greedy_b)
check("避免局部贪心错位（取全局交集 7s 的合并配对）",
      pairing(rows) == [
          ([0], [], "a_only"), ([1], [0, 1], "pair")]
      and rows[1].overlap == 7.0)

# 同一组配对交集总时长相同的平局：优先不合并
# A1=[0,2) A2=[2,4)；B1=[0,2) B2=[2,4) —— 两个一对一交集各 2（总 4），
# A1+A2 与任何单条 B 都拿不到 4，因此两个一对一必然最优
tie_rows = align(
    [Cue(0, 2, "a1"), Cue(2, 4, "a2")],
    [Cue(0, 2, "b1"), Cue(2, 4, "b2")])
check("平局优先合并次数最少（保持两个一对一）",
      all(len(r.a) == len(r.b) == 1 for r in tie_rows))

# 平局再平局时选中心偏差更小的：A1=[0,4) 只可能与 B1/B2 之一正交集，
# 构造 B1=[0,4)（dev=0）与 B2=[4,8)（不重叠），结果唯一
center_rows = align([Cue(0, 4, "a1")],
                    [Cue(0, 4, "b1"), Cue(4, 8, "b2")])
check("正交集配对、零交集独立成行", pairing(center_rows) == [
    ([0], [0], "pair"), ([], [1], "b_only")])

# 相邻 cue 间有间隙：合并组按时间并集（含间隙）计分
# A1=[0,2) A2=[5,7)（中间 3s 间隙）；B1=[1,6)
# 合并 A1+A2 并集 [0,7) ∩ [1,6) = 5，优于 A1 单独交集 1
gap_rows = align(
    [Cue(0, 2, "a1"), Cue(5, 7, "a2")],
    [Cue(1, 6, "b1"), Cue(7, 9, "b2")])
check("合并组按时间并集（含中间间隙）计分",
      pairing(gap_rows) == [
          ([0, 1], [0], "pair"), ([], [1], "b_only")]
      and gap_rows[0].overlap == 5.0)

# 仅端点相接不算正时长交集 → 不配对
touch_rows = align([Cue(0, 2, "a1")], [Cue(2, 4, "b1")])
check("端点相接（零时长交集）不配对",
      kinds(touch_rows) == ["a_only", "b_only"])

# 完全错开的时间轴：所有 cue 独立成行
apart_rows = align([Cue(0, 1, "a1")], [Cue(5, 6, "b1")])
check("时间轴完全错开时全部独立成行",
      pairing(apart_rows) == [
          ([0], [], "a_only"), ([], [0], "b_only")])

# 行顺序沿时间轴单调：三段互不重叠（配对在前，两侧独立行跟随）
order_rows = align(
    [Cue(0, 2, "a1"), Cue(10, 12, "a2")],
    [Cue(0, 2, "b1"), Cue(20, 22, "b2")])
check("未配对 cue 独立成行且顺序单调",
      pairing(order_rows) == [
          ([0], [0], "pair"), ([1], [], "a_only"),
          ([], [1], "b_only")])

# 空输入
check("空对照轨：修订 cue 全部独立",
      [r.kind for r in align([Cue(0, 1, "a")], [])] == ["a_only"])
check("空修订轨：对照 cue 全部独立",
      [r.kind for r in align([], [Cue(0, 1, "b")])] == ["b_only"])
check("两轨皆空得到零行", align([], []) == [])

# 规模与一致性：每侧每条 cue 恰好出现一次
big_a = [Cue(i, i + 1, f"a{i}") for i in range(300)]
big_b = [Cue(i, i + 1, f"b{i}") for i in range(250)]  # B 少 50 条
big_rows = align(big_a, big_b)
used_a = sorted(k for r in big_rows for k in r.a)
used_b = sorted(k for r in big_rows for k in r.b)
check("大规模对齐中 A 侧每 cue 恰好出现一次", used_a == list(range(300)))
check("大规模对齐中 B 侧每 cue 恰好出现一次", used_b == list(range(250)))
check("大规模对齐前 250 条逐对配对",
      all(r.kind == "pair" for r in big_rows[:250]))

print("ALIGNER_OK")
