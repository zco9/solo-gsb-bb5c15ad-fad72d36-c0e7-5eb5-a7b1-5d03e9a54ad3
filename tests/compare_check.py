"""双轨对照对齐纯逻辑校验：

    python3 tests/compare_check.py

仅依赖标准库（直接导入 app.compare / app.subtitles）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.compare import align_tracks  # noqa: E402
from app.subtitles import Cue  # noqa: E402


def C(start: float, end: float, text: str = "") -> Cue:
    return Cue(start=start, end=end, text=text)


def kinds(result):
    out = []
    for r in result.rows:
        if not r.right:
            out.append("L")
        elif not r.left:
            out.append("R")
        elif len(r.left) == 2:
            out.append("2L")
        elif len(r.right) == 2:
            out.append("2R")
        else:
            out.append("11")
    return out


def check(name, cond):
    assert cond, name
    print("ok -", name)


def main() -> None:
    # 1. 完全一致：全部 1:1，交集满、零偏差、无合并
    res = align_tracks(
        [C(0, 1), C(1, 2), C(2, 3)],
        [C(0, 1), C(1, 2), C(2, 3)])
    check("相同轨道全部 1:1", kinds(res) == ["11", "11", "11"])
    check("相同轨道交集总时长 3000ms", res.total_overlap_ms == 3000)
    check("相同轨道无合并", res.merge_count == 0)
    check("相同轨道零中心偏差", res.total_center_delta_ms == 0)
    check("统计：无独立行",
          (res.left_only_rows, res.right_only_rows) == (0, 0))

    # 2. 仅端点相接不算正时长交集：各自独立成行（顺序由 DP tie-break 决定）
    res = align_tracks([C(0, 1)], [C(1, 2)])
    check("端点相接不配对",
          set(kinds(res)) == {"L", "R"} and res.paired_rows == 0)

    # 3. 完全不相交：独立成行
    res = align_tracks([C(0, 1)], [C(5, 6)])
    check("不相交不配对", res.total_overlap_ms == 0 and res.paired_rows == 0)
    check("不相交各自成行",
          res.left_only_rows == 1 and res.right_only_rows == 1)

    # 4. 2:1：左轨两条相邻 cue（含间隙）合并，交集按时间并集计分
    res = align_tracks([C(0, 1), C(2, 3)], [C(0, 3)])
    check("左二对一采用合并配对", kinds(res) == ["2L"])
    check("并集交集 3000ms（跨间隙）", res.total_overlap_ms == 3000)
    check("合并次数为 1", res.merge_count == 1)

    # 5. 1:2：右轨两条相邻 cue 合并
    res = align_tracks([C(0, 3)], [C(0, 1), C(2, 3)])
    check("一对右二采用合并配对", kinds(res) == ["2R"])
    check("右合并并集交集 3000ms", res.total_overlap_ms == 3000)

    # 6. 合并次数最少优先：两个 1:1 与一次 2:1 交集相同时选 1:1
    #    L1[0,1]+L2[1,2] 并集 [0,2] 与 R[0,2] 交集 2000；
    #    但分别 1:1 也是 2000 且无合并 → 必须选两个 1:1
    res = align_tracks([C(0, 1), C(1, 2)], [C(0, 1), C(1, 2)])
    check("交集相同时合并次数最少", kinds(res) == ["11", "11"])
    check("该解合并次数为 0", res.merge_count == 0)

    # 7. 有序对齐不错位：L1 主要搭 R1、L2 主要搭 R2 时保持一一对应
    res = align_tracks([C(0, 3), C(3, 6)], [C(2, 3.1), C(3.1, 6)])
    check("顺序对齐不错位", kinds(res) == ["11", "11"])
    check("交集总和 3900ms", res.total_overlap_ms == 3900)

    # 8. 全局 DP 优于顺序贪心：
    #    R1[1,4.5] 横跨 L1[0,3]/L2[3,6] 边界，R2[0,1.05] 只搭 L1 一点。
    #    顺序贪心会先把 L1-R1（2000ms）配掉，之后 L2 无 cue 可配、总分 2000；
    #    全局 DP 把 L1+L2 合并（并集 [0,6]）配 R1 得 3500ms，R2 独立成行。
    res = align_tracks([C(0, 3), C(3, 6)],
                       [C(1, 4.5), C(0, 1.05)])
    check("跨边界长 cue 采用合并配对",
          [(r.left, r.right) for r in res.rows] == [([0, 1], [0]), ([], [1])])
    check("合并配对按并集交集 3500ms", res.total_overlap_ms == 3500)
    check("该解合并 1 次、右独立 1 行",
          res.merge_count == 1 and res.right_only_rows == 1)

    #    常规有序错位防护：L1 与 R1 小交、L2 与 R2 大交时保持有序 1:1
    res = align_tracks([C(0, 2.05), C(2.05, 3)],
                       [C(0, 2), C(2, 3)])
    check("有序情形保持 1:1", kinds(res) == ["11", "11"])
    check("有序情形交集总和 2950ms", res.total_overlap_ms == 2950)

    # 9. 中心偏差 tie-break：交集方案相同时选中心更近的组合
    #    L1[0,4] L2[4,8]；R1[0,4](与 L1 中心齐) R2[4,8](与 L2 中心齐)。
    #    交叉配（L1-R2、L2-R1）无正交集，只能正配，偏差应为 0。
    res = align_tracks([C(0, 4), C(4, 8)], [C(0, 4), C(4, 8)])
    check("中心对齐时偏差为 0", res.total_center_delta_ms == 0)

    #    真正的 tie：单 cue 对两个交集相同的候选，由 DP 全局决定；
    #    L=[0,4],[4,8]，R=[0,3.9],[3.9,8]：
    #    正配交集 3.9+4=7.9s；L1-R2 交 .1s 且 L2-R1 无交集 → 正配更优。
    res = align_tracks([C(0, 4), C(4, 8)], [C(0, 3.9), C(3.9, 8)])
    check("交集主导选正配", kinds(res) == ["11", "11"])
    # 正配偏差：|2-1.95|*1000=50，|6-5.95|*1000=50 → 100ms
    check("正配中心偏差 100ms", res.total_center_delta_ms == 100)

    # 10. 空轨：全部独立成行
    res = align_tracks([], [C(0, 1), C(1, 2)])
    check("空左轨独立两行", kinds(res) == ["R", "R"])
    res = align_tracks([C(0, 1)], [])
    check("空右轨独立一行", kinds(res) == ["L"])

    # 11. 每行序号覆盖全部 cue 且不重复（对齐是划分）
    res = align_tracks(
        [C(i, i + 1.2) for i in range(5)],
        [C(i + 0.1, i + 1.0) for i in range(6)])
    seen_l, seen_r = [], []
    for r in res.rows:
        seen_l += r.left
        seen_r += r.right
    check("左轨 cue 恰好覆盖一次", sorted(seen_l) == list(range(5)))
    check("右轨 cue 恰好覆盖一次", sorted(seen_r) == list(range(6)))

    # 12. 行序按时间（左索引单调不减；首个出现的右索引也单调）
    firsts = [r.left[0] for r in res.rows if r.left]
    check("对照行按左轨时间排序", firsts == sorted(firsts))

    print("COMPARE_CHECK_OK")


if __name__ == "__main__":
    main()
