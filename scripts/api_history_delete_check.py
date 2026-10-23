"""api_history_delete_check.py —— 「对局记录删除」端到端闸门

用户需求（2026-10-12）
---------------------
* 对局记录面板每行有**删除按钮**；
* **数据统计仅统计对局记录里的对局数据** —— 删掉记录，统计要跟着掉；
* 删掉一局时，它的**回放**也要一并消失（不再能点开）。

做法
----
起一个真 `web/dushan_server.py`，但把历史目录用 ``DUSHAN_HISTORY_DIR``
指到**临时目录**，往里塞几条合成的 records.jsonl / replays.jsonl ——
这样整条链路跑在真 HTTP 上，又**完全不动** ``web/history/`` 里用户的真实战绩。

覆盖：/api/history 的 _ln 定位 / has_replay 标记 → /api/stats 基线 →
       负例（rid 不匹配、行号越界）→ 正例删除 → 统计同步 -1 →
       记录列表消失 → /api/replay 打不开 → 静态页仍能正常加载。

坑（踩过一次，记下来）
--------------------
* 服务端 **bind 完就打印横幅、随后才进 serve_forever**，所以启动瞬间的首个
  ``/api/history`` 有极小概率拿到空表 —— 脚本用「轮询到看见本次 MARK」做热身。
* 端口不要写死：上次跑剩下的 TIME_WAIT / 残留进程会让新实例 bind 到同一个端口，
  于是探活打到了**上一次的子进程**（它读的历史目录已被删 → 全空）。
  这里默认用 ``free_port()`` 取空闲端口，并用随机 ``MARK`` 校验「答话的就是本次的子进程」。

跑法::

    python scripts/api_history_delete_check.py [端口]
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    """挑一个当前空闲的端口 —— 避免上一次跑剩的 TIME_WAIT / 残留进程把端口占住，
    导致「探活打到了别的服务」这种假失败。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else free_port()
BASE = f"http://127.0.0.1:{PORT}"

ok = 0
bad = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, bad
    if cond:
        ok += 1
    else:
        bad += 1
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def call(path: str, body: dict | None = None, timeout: float = 30.0):
    """POST/GET 一个 JSON 接口。拒删时服务端按约定回 **400 + 结构化错误体**，
    这里把 4xx/5xx 的响应体也照常解析出来（否则测不到「拒删」这类分支）。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}"}


def get_raw(path: str, timeout: float = 30.0):
    """GET 原始响应 ``(status, headers, body_bytes)`` —— 导出接口要验 BOM 与
    ``Content-Disposition``，走 JSON 化的 ``call`` 看不出来。"""
    req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


# ---- 合成记录：3 条（两条 type=win + 一条 abort）----
#   my_seat 一律 0；deltas 零和；只有 win 的局计入 games（abort 不计任何一项）。
#   rid 里带一个随机标记（MARK）—— 用来确认「答话的服务端就是我们的子进程」：
#   若端口上恰好有别的 dushan_server 在跑，它读的是别的历史目录，看不到 MARK。
MARK = uuid.uuid4().hex[:8]
RID_A, RID_B, RID_C = f"rid-a-{MARK}", f"rid-b-{MARK}", f"rid-c-{MARK}"


def seed(tmp: Path) -> tuple[list[dict], dict]:
    recs = [
        {"rid": RID_A, "type": "win", "my_seat": 0, "is_tsumo": True,
         "deltas": [9.0, -3.0, -3.0, -3.0], "winner": 0, "winners": [0],
         "fan_cn": "平胡", "winners_fan": [{"fan_cn": "平胡"}],
         "flags": {"win_bigfan": False, "huang": False}, "detail": []},
        {"rid": RID_B, "type": "win", "my_seat": 0, "is_tsumo": False,
         "deltas": [-6.0, 12.0, -3.0, -3.0], "winner": 1, "winners": [1],
         "fan_cn": "清一色", "winners_fan": [{"fan_cn": "清一色"}],
         "flags": {"win_bigfan": True, "huang": False}, "detail": []},
        {"rid": RID_C, "type": "abort", "my_seat": 0,
         "deltas": [0.0, 0.0, 0.0, 0.0], "detail": []},
    ]
    (tmp / "records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
        encoding="utf-8")
    reps = {
        RID_A: {"rid": RID_A, "hand": 1, "my_seat": 0, "type": "win",
                "deal": [[0], [0], [0], [0]],
                "ev": [[1, "draw", 0, 0, "", 1, [1, 0, 0, 0], "x", [0, 0, 0, 0]]]},
        RID_B: {"rid": RID_B, "hand": 2, "my_seat": 0, "type": "win",
                "deal": [[0], [0], [0], [0]],
                "ev": [[1, "draw", 0, 0, "", 1, [1, 0, 0, 0], "x", [0, 0, 0, 0]]]},
    }
    (tmp / "replays.jsonl").write_text(
        "".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n"
                for v in reps.values()), encoding="utf-8")
    return recs, reps


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dushan_histdel_"))
    log = ROOT / "reports" / f"_api_histdel_server_{PORT}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    seed(tmp)
    print(f"隔离历史目录：{tmp}")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["DUSHAN_HISTORY_DIR"] = str(tmp)

    print(f"启动服务 {BASE} …", flush=True)
    srv = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "web" / "dushan_server.py"),
         "--host", "127.0.0.1", "--port", str(PORT), "--threads", "4"],
        cwd=str(ROOT), env=env, stdout=open(log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                call("/api/meta", timeout=5)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("服务没起来，看", log)
            print(log.read_text(encoding="utf-8")[-2000:])
            return 1

        # 身份校验 + 热身：确认答话的确实是**我们刚起的子进程**（读的是隔离历史目录）。
        # 服务端刚 bind 完就打印横幅、随后才进 serve_forever，**首个 /api/history
        # 有极小概率拿到空表**（启动竞态），所以这里轮询到「看见本次 MARK」为止再开始断言。
        probe: list[dict] = []
        for _ in range(40):
            probe = call("/api/history?limit=50").get("records") or []
            if any(str(x.get("rid") or "") == RID_A for x in probe):
                break
            time.sleep(0.25)
        if srv.poll() is not None or not any(
                str(x.get("rid") or "") == RID_A for x in probe):
            print("  ✗ 端口上的服务端不是本次的子进程（换端口重跑）："
                  f"child_alive={srv.poll() is None}，"
                  f"history 命中 MARK={any(str(x.get('rid') or '') == RID_A for x in probe)}")
            print(f"  诊断：BASE={BASE} tmp={tmp} pid={srv.pid} 日志字节={log.stat().st_size}"
                  f" history={len(probe)} 条 {[x.get('rid') for x in probe[:3]]}")
            print("  --- 服务端日志 ---")
            print(log.read_text(encoding="utf-8")[-1500:])
            return 1

        # ---- 1) 记录列表：_ln 定位 + has_replay ----
        hist = call("/api/history?limit=30")
        rows = hist.get("records") or []
        if len(rows) != 3:      # 环境问题（端口被占/历史目录没生效）→ 直接把日志打出来
            print("  ！记录数不是 3，先把服务端日志贴出来：")
            print(log.read_text(encoding="utf-8")[-1500:], flush=True)
        check("GET /api/history 返回全部 3 条记录", len(rows) == 3, f"{len(rows)} 条")
        check("列表按最新在前（rid-c 在最前）",
              bool(rows) and rows[0].get("rid") == RID_C,
              str([r.get("rid") for r in rows]))
        check("每条都带 0 基行号 _ln（删除定位用）",
              sorted(int(r["_ln"]) for r in rows if "_ln" in r) == [0, 1, 2],
              str([r.get("_ln") for r in rows]))
        by_rid = {r.get("rid"): r for r in rows}
        check("有回放的局标记 has_replay=True", by_rid[RID_A].get("has_replay") is True)
        check("没有回放的局标记 has_replay=False", by_rid[RID_C].get("has_replay") is False)
        # 删之前两条回放都应能打开（下面「删 A 后 B 仍可打开」要有基线才有意义）
        for rid in (RID_A, RID_B):
            rp = call(f"/api/replay?rid={rid}")
            check(f"回放 {rid.split('-')[1]} 可打开",
                  bool(rp.get("ok")) and (rp.get("replay") or {}).get("rid") == rid,
                  str(rp)[:80])

        # ---- 2) 统计基线：abort 不计入 ----
        s0 = (call("/api/stats").get("all_time") or {})
        check("统计基线 games = 2（abort 的 rid-c 不计）", s0.get("games") == 2,
              f"games={s0.get('games')}")
        check("统计基线 win_bigfan_games = 1（rid-b 清一色）",
              s0.get("win_bigfan_games") == 1, f"{s0.get('win_bigfan_games')}")

        # ---- 3) 负例：rid 与行号不匹配 → 拒删 ----
        ln_a = int(by_rid[RID_A]["_ln"])
        r = call("/api/history_delete", {"ln": ln_a, "rid": RID_B})
        check("rid 与行号不匹配 → 拒删", (not r.get("ok")) and "不匹配" in str(r.get("error")),
              str(r))
        check("拒删后记录条数不变", len((call("/api/history?limit=30").get("records") or [])) == 3)
        check("拒删后统计不变", (call("/api/stats").get("all_time") or {}).get("games") == 2)

        # ---- 4) 负例：行号越界 → 拒删 ----
        r = call("/api/history_delete", {"ln": 99, "rid": RID_A})
        check("行号越界 → 拒删", (not r.get("ok")) and "越界" in str(r.get("error")), str(r))

        # ---- 5) 正例：删 rid-a（有回放的那条）----
        r = call("/api/history_delete", {"ln": ln_a, "rid": RID_A})
        check("POST /api/history_delete 删成功（removed_rec=1, removed_replay=1）",
              bool(r.get("ok")) and r.get("removed_rec") == 1 and r.get("removed_replay") == 1,
              str(r))

        # ---- 6) 统计随记录同步（「数据统计仅统计对局记录里的对局数据」）----
        s1 = (call("/api/stats").get("all_time") or {})
        check("删一条记录后 games 恰好 -1（2 → 1）", s1.get("games") == 1,
              f"{s0.get('games')} → {s1.get('games')}")

        rows2 = call("/api/history?limit=30").get("records") or []
        check("被删记录不再出现在列表里",
              [x.get("rid") for x in rows2] == [RID_C, RID_B],
              str([x.get("rid") for x in rows2]))
        check("行号已被重排（rid-c 仍在、_ln 重新连续）",
              sorted(int(x["_ln"]) for x in rows2) == [0, 1],
              str([x.get("_ln") for x in rows2]))

        # ---- 7) 回放一并消失 ----
        rp = call(f"/api/replay?rid={RID_A}")
        check("被删局的回放打不开（/api/replay 报已不在缓存）", not rp.get("ok"), str(rp)[:90])
        rp2 = call(f"/api/replay?rid={RID_B}")
        check("其余局的回放仍可打开", bool(rp2.get("ok")) and rp2.get("replay", {}).get("rid") == RID_B,
              str(rp2)[:90])

        # ---- 8) 删掉最后一条 → games = 0；再删应报「已不在文件里」 ----
        ln_b = int([x for x in rows2 if x.get("rid") == RID_B][0]["_ln"])
        r = call("/api/history_delete", {"ln": ln_b, "rid": RID_B})
        check("再删一条（rid-b）成功", bool(r.get("ok")), str(r))
        s2 = (call("/api/stats").get("all_time") or {})
        check("删第二条后 games = 0", s2.get("games") == 0, f"{s2.get('games')}")
        check("win_bigfan_games 也归零（口径跟随记录）",
              s2.get("win_bigfan_games") == 0, f"{s2.get('win_bigfan_games')}")
        r = call("/api/history_delete", {"ln": 0, "rid": RID_A})
        check("重复删除同一 rid → 被拒（不匹配 / 已不在文件里 / 越界）",
              (not r.get("ok")) and any(x in str(r.get("error"))
                                        for x in ("不匹配", "已不在文件里", "越界")), str(r))
        check("重复删除后统计不变（games 仍为 0）",
              (call("/api/stats").get("all_time") or {}).get("games") == 0)

        # ---- 9) 批量删除 + 导出（2026-10-13 用户需求新增）----
        #   ① 批量：{"lns":[…]}/{"rids":[…]} 逐项尽力删，能删的删、删不了的进 skipped；
        #   ② 导出：/api/history_export?format=json|csv 直接下载（带 Content-Disposition）。
        # 这里重新种 5 条（rid 换一批前缀，避免和上面已删的撞缓存）。
        M2 = uuid.uuid4().hex[:8]
        b_recs = [
            {"rid": f"b{i}-{M2}", "type": "win", "my_seat": 0, "is_tsumo": True,
             "deltas": [3.0, -1.0, -1.0, -1.0], "winner": 0, "winners": [0],
             "fan_cn": "平胡", "flags": {"win_bigfan": False}, "detail": []}
            for i in range(5)
        ]
        (tmp / "records.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in b_recs),
            encoding="utf-8")
        (tmp / "replays.jsonl").write_text(
            "".join(json.dumps({"rid": r["rid"], "hand": 1, "my_seat": 0,
                                "type": "win", "deal": [[0], [0], [0], [0]],
                                "ev": []}, ensure_ascii=False,
                               separators=(",", ":")) + "\n" for r in b_recs),
            encoding="utf-8")
        rows3 = call("/api/history?limit=30").get("records") or []
        check("重新种 5 条 → _ln 连续 0..4",
              [int(x["_ln"]) for x in rows3] == [4, 3, 2, 1, 0],
              str([x.get("_ln") for x in rows3]))

        # 批量删「行号 1 + 行号 3」两条（各带 rid 做安全校验）
        rids_by_ln = {int(x["_ln"]): x["rid"] for x in rows3}
        r = call("/api/history_delete",
                 {"lns": [1, 3], "rids": [rids_by_ln[1], rids_by_ln[3]]})
        left_rids = [x.get("rid") for x in (call("/api/history?limit=30").get("records") or [])]
        check("批量删 2 条 → ok / removed_rec=2 / removed_replay=2",
              bool(r.get("ok")) and r.get("removed_rec") == 2 and r.get("removed_replay") == 2,
              str(r))
        check("批量删后只剩另 3 条（行号 1/3 的两条消失）",
              left_rids == [rids_by_ln[4], rids_by_ln[2], rids_by_ln[0]],
              str(left_rids))
        check("批量删的行号已重排 _ln 0..2",
              sorted(int(x["_ln"]) for x in
                     (call("/api/history?limit=30").get("records") or [])) == [0, 1, 2])

        # 批量里混一个越界行号：能删的照删，删不了的进 skipped（不整批失败）
        # ⚠️ 行号在上一批删除后已重排，必须按**当前** _ln 取，不能沿用删除前的。
        cur = call("/api/history?limit=30").get("records") or []
        ln0, rid0 = int(cur[0]["_ln"]), cur[0]["rid"]
        r = call("/api/history_delete", {"lns": [ln0, 99], "rids": [rid0, "nope"]})
        check("批量混入越界行号 → 部分成功（removed_rec=1，skipped=1）",
              bool(r.get("ok")) and r.get("removed_rec") == 1 and len(r.get("skipped") or []) == 1,
              str(r))
        check("skipped 里带可读原因（含「越界」）",
              "越界" in str((r.get("skipped") or [{}])[0].get("why", "")), str(r.get("skipped")))

        # 只按 rids 批量删剩下的 2 条
        rest = [x["rid"] for x in (call("/api/history?limit=30").get("records") or [])]
        r = call("/api/history_delete", {"rids": rest})
        check("只按 rids 批量删 → 清空",
              bool(r.get("ok")) and r.get("removed_rec") == len(rest) and
              not (call("/api/history?limit=30").get("records") or []),
              str(r))

        # ---- 导出：重新种 3 条，验 JSON / CSV 两个格式 ----
        (tmp / "records.jsonl").write_text(
            "".join(json.dumps(r_, ensure_ascii=False) + "\n" for r_ in b_recs[:3]),
            encoding="utf-8")
        st_, hd, body = get_raw("/api/history_export?format=json")
        j = json.loads(body.decode("utf-8"))
        check("导出 JSON：count=3 且 records 长度一致",
              st_ == 200 and j.get("count") == 3 and len(j.get("records") or []) == 3,
              f"status={st_} count={j.get('count')}")
        check("导出 JSON：带 attachment 文件名 (*.json)",
              "attachment" in hd.get("Content-Disposition", "") and
              hd.get("Content-Disposition", "").endswith('.json"'),
              hd.get("Content-Disposition", ""))
        st_, hd, body = get_raw("/api/history_export?format=csv")
        text = body.decode("utf-8")
        lines = [x for x in text.splitlines() if x.strip()]
        check("导出 CSV：带 UTF-8 BOM（Excel 中文不乱码）", text.startswith("\ufeff"),
              repr(text[:8]))
        check("导出 CSV：表头 20 列且含中文列名",
              len(lines[0].split(",")) == 20 and "时间" in lines[0] and "四家分" in lines[0],
              f"{len(lines[0].split(','))} 列")
        check("导出 CSV：数据 3 行（+表头 1 行）", len(lines) == 4, f"n={len(lines)}")
        # 导出是只读的：导完记录还是 3 条
        check("导出不改动记录文件",
              len(call("/api/history?limit=30").get("records") or []) == 3)

        # ---- 10) 静态页不受影响 ----
        with urllib.request.urlopen(BASE + "/", timeout=10) as resp:
            html = resp.read().decode("utf-8")
        check("GET / 仍返回牌桌页面", resp.status == 200 and "麻将" in html, f"{len(html)} 字节")

    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n合计 {ok + bad} 项，失败 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
