#!/usr/bin/env python3
"""把「测试通过的更强模型」登记为线上**附加**模型 —— 只增不替（2026-11-07 用户口径）。

纪律（写死在工具里，防止手滑）：
  1. 绝不写 models/dushan_master.pt（它同时是闸门 A/C 的冻结基线）；
  2. 绝不改 MASTER_CANDIDATES / 前端默认档位 —— 线上主力恒为原 master；
  3. 只在 models/champions.json 追加一条，并在 models/ 下新增一个 .pt；
     已有同名权重一律拒绝覆盖（只允许 --in-place 登记已有文件）；
  4. 服务端 /api/meta 下发「主力 + 全部冠军」，前端动态追加下拉选项。

用法：
  # 登记训练产物（复制成 models/champ_exit_p20.pt，不覆盖任何已有文件）
  .venv/bin/python tools/register_champion.py --src models/exit_p20.pt \
      --label "冠军·P20" --gate "gateD +0.212±0.330 (ns)；仅登记不晋升"
  # 登记磁盘上已有的模型（不复制）
  .venv/bin/python tools/register_champion.py --src models/web_best.pt --in-place \
      --label "ExIt 学生（R07 快照）"
  # 查看注册表
  .venv/bin/python tools/register_champion.py --list
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REGISTRY = MODELS / "champions.json"
FORBIDDEN = {"dushan_master", "master", "best", "rl"}


def load_rows() -> list[dict]:
    if not REGISTRY.exists():
        return []
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        rows = data.get("champions", []) if isinstance(data, dict) else []
        return [r for r in rows if isinstance(r, dict)]
    except (OSError, ValueError):
        return []


def save_rows(rows: list[dict]) -> None:
    REGISTRY.write_text(
        json.dumps({"_note": "线上附加冠军模型：只增不替；主力恒为 dushan_master.pt",
                    "champions": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="源 .pt 权重路径")
    ap.add_argument("--stem", default="", help="目标 stem（默认 champ_<源文件名>）")
    ap.add_argument("--label", default="", help="前端展示名")
    ap.add_argument("--gate", default="", help="通过依据（闸门/复赛结论）")
    ap.add_argument("--note", default="", help="备注")
    ap.add_argument("--in-place", action="store_true",
                    help="源已在 models/ 下，直接用原名登记，不复制")
    ap.add_argument("--list", action="store_true", help="只打印注册表")
    a = ap.parse_args()

    if a.list:
        for i, r in enumerate(load_rows(), 1):
            print(f"{i:2d}. {r.get('stem')}  {r.get('label', '')}  "
                  f"added={r.get('added', '')}  gate={r.get('gate', '')}")
        print(f"共 {len(load_rows())} 个附加模型；线上主力恒为 dushan_master")
        return 0

    if not a.src:
        ap.error("需要 --src（或用 --list）")
    src = Path(a.src)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        print(f"[FAIL] 源权重不存在：{src}", file=sys.stderr)
        return 1

    stem = a.stem or (src.stem if a.in_place else f"champ_{src.stem}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stem):
        print(f"[FAIL] 非法 stem：{stem}", file=sys.stderr)
        return 1
    if stem in FORBIDDEN or stem.startswith("dushan_master"):
        print(f"[FAIL] {stem} 是受保护的主力名，禁止登记（只增不替）", file=sys.stderr)
        return 1
    if a.in_place and src.parent.resolve() != MODELS.resolve():
        print("[FAIL] --in-place 要求源就在 models/ 下", file=sys.stderr)
        return 1

    dst = MODELS / f"{stem}.pt"
    if a.in_place:
        if dst != src:
            print(f"[FAIL] --in-place 时 --stem 必须等于源文件名（{src.stem}）",
                  file=sys.stderr)
            return 1
        action = "就地登记（未复制、未改动权重）"
    else:
        if dst.exists():
            print(f"[FAIL] {dst.name} 已存在，拒绝覆盖（只增不替）；"
                  "如确要登记它请改用 --in-place", file=sys.stderr)
            return 1
        shutil.copy2(src, dst)
        action = f"复制 {src.name} → {dst.name}"

    rows = load_rows()
    hit = next((r for r in rows if r.get("stem") == stem), None)
    rec = {
        "stem": stem,
        "label": a.label or stem,
        "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "gate": a.gate,
        "note": a.note,
        "added": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if hit is None:
        rows.append(rec)
        verb = "新增"
    else:
        hit.update({k: v for k, v in rec.items() if v})
        verb = "更新登记信息"
    save_rows(rows)
    print(f"[OK] {verb} {stem}：{action}；标签「{rec['label']}」")
    print(f"     注册表 {REGISTRY}（共 {len(rows)} 个附加模型）")
    print("     线上主力仍为 dushan_master；前端刷新后可在模型下拉里选中该冠军")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
