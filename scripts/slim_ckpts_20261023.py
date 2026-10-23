# -*- coding: utf-8 -*-
"""D 阶段磁盘瘦身：_ckpts/ 与 reports/archive/*/ckpts 保留策略。

保留（keep set）：
  1. 所有 judge json / 报告 json 中引用过的检查点 stem（判读证据点）
  2. 每个运行前缀的最后一个检查点（可续训终点）
  3. dushan_m2_c4_it2280（白名单最强学生，显式保护）

删除：
  - models/_ckpts/ 中其余检查点
  - reports/archive/*/ckpts/ 中不在 keep set 的硬链接
  （MANIFEST / judge json / history / out / csv 等证据文件一律不动）

先写 manifest CSV（含路径/大小/动作）再删除，可审计。
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "models", "_ckpts")
ARCHIVE = os.path.join(ROOT, "reports", "archive")
MANIFEST = os.path.join(ROOT, "reports", "_ckpt_slim_manifest_20261023.csv")

PROTECTED = {"dushan_m2_c4_it2280.pt"}
STEM_RE = re.compile(
    r"dushan_(?:m1_s\d+|m2_[a-z0-9]+|big_s\d+|b2[a-z_]*?|b2c_[a-z_]+?)_it\d+"
)


def collect_referenced():
    """从 reports/*.json 里抓所有引用过的检查点 stem。"""
    refs = set()
    for fn in os.listdir(os.path.join(ROOT, "reports")):
        if not fn.endswith(".json") or fn.startswith("_ckpt_slim"):
            continue
        p = os.path.join(ROOT, "reports", fn)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        for m in STEM_RE.finditer(text):
            refs.add(m.group(0))
    return refs


def per_prefix_last(files):
    """每个运行前缀（去掉 _itXXXX）的最后（最大 it）一个文件名。"""
    last = {}
    for fn in files:
        m = re.match(r"(.+)_it(\d+)\.pt$", fn)
        if not m:
            continue
        pref, it = m.group(1), int(m.group(2))
        if pref not in last or it > last[pref][0]:
            last[pref] = (it, fn)
    return {fn for _, fn in last.values()}


def main(apply: bool):
    refs = collect_referenced()
    ckpts = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pt"))
    keep_ckpts = {f for f in ckpts
                  if f in PROTECTED or f[:-3] in refs} | per_prefix_last(ckpts)
    del_ckpts = [f for f in ckpts if f not in keep_ckpts]

    rows = []
    freed = 0
    for f in del_ckpts:
        p = os.path.join(CKPT_DIR, f)
        sz = os.path.getsize(p)
        rows.append(("delete", os.path.relpath(p, ROOT), sz))
        freed += sz

    # archive 侧：只删 ckpts/ 里不在 keep 的硬链接
    arch_del = []
    if os.path.isdir(ARCHIVE):
        for run in sorted(os.listdir(ARCHIVE)):
            cdir = os.path.join(ARCHIVE, run, "ckpts")
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.endswith(".pt") and f not in keep_ckpts:
                    p = os.path.join(cdir, f)
                    sz = os.path.getsize(p)
                    # 硬链接：只有当 _ckpts 侧同文件也被删（或已不存在）时才真正释放
                    twin = os.path.join(CKPT_DIR, f)
                    real = sz if not os.path.exists(twin) else 0
                    arch_del.append(p)
                    rows.append(("delete", os.path.relpath(p, ROOT), sz))
                    freed += real

    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["action", "path", "size_bytes"])
        w.writerows(rows)

    n_keep = len(keep_ckpts)
    print(f"keep(_ckpts)={n_keep}/{len(ckpts)}  del(_ckpts)={len(del_ckpts)}"
          f"  del(archive ckpts)={len(arch_del)}")
    print(f"referenced stems: {len(refs)}")
    print(f"estimated freed: {freed / 2**30:.2f} GiB")
    print("kept points:", ", ".join(sorted(keep_ckpts)[:40]))

    if not apply:
        print("DRY-RUN：未删除任何文件。manifest ->", MANIFEST)
        return 0

    ndel = 0
    for _, rel, _ in rows:
        p = os.path.join(ROOT, rel)
        try:
            os.remove(p)
            ndel += 1
        except OSError as e:
            print("FAIL", rel, e, file=sys.stderr)
    print(f"APPLIED: deleted {ndel} files. manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
