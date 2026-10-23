"""探针：验证 --klt-teacher 真的把「蒸馏教师」与「一等对手座位」拆开了。

为什么需要它
------------
m2c4–m2c5 的 +0.386 无法归因，根因是 ``--master-ckpt`` 一物两用：
同一个实例既算 KL 教师，又占 0.40 权重的对手座。改造后这两条路径各走各的
worker 缓存（``_TEACH_CACHE`` / ``_MASTER_CACHE``），但仍可能在某处又合并回去
（例如 shared 分支写错、缓存 key 串了）。

所以这里不看日志文案，直接 **spy worker 内部拿到的两个对象**，比较参数量与
对象身份。用两个参数量明显不同的模型做区分：

* 教师 = ``_ckpts/dushan_m2_d3_it0020``（9.84M，v3）
* 座位 = ``dushan_master``（767k，v1）

用法::

    python scripts/_probe_klt_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import zhuoji.vcollect as V                                    # noqa: E402
from zhuoji.net import build_from_ckpt, ck_from_model          # noqa: E402

REC: dict = {}
_orig = V.collect_vectorized


def _spy(*a, **kw):
    t = kw.get("teacher")
    m = kw.get("master_model")
    REC["teacher_n"] = int(t.n_params()) if t is not None else None
    REC["master_n"] = int(m.n_params()) if m is not None else None
    REC["same_obj"] = (t is m)
    return _orig(*a, **kw)


def build(name: str):
    ck = torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu",
                    weights_only=False)
    m, v = build_from_ckpt(ck)
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, v, ck["net"], ck.get("encoder_packs")


def run(shared: bool) -> dict:
    main, main_v, nc, main_pk = build("_ckpts/dushan_m2_c5_it3120")
    klt_m, klt_v, _, _ = build("_ckpts/dushan_m2_d3_it0020")
    mas_m, mas_v, _, _ = build("dushan_master")
    klt_ck = ck_from_model(klt_m, encoder_version=klt_v)
    mas_ck = ck_from_model(mas_m, encoder_version=mas_v)

    V._TEACH_CACHE["model"] = None
    V._MASTER_CACHE["model"] = None
    V.collect_parallel(
        main.state_dict(), nc, 8, 1, 1.0, 8.0, 0.05, workers=1,
        past_prob=0.0, snapshots_sd=[], rules="dushan", want_traj=False,
        encoder_version=main_v, encoder_packs=main_pk, snap_key=None,
        explore_eps=0.0,
        teacher_ck=klt_ck, teacher_ver=klt_v, master_prob=1.0,
        master_ck=mas_ck, master_ver=mas_v, master_shared=shared)
    return dict(REC)


def main() -> int:
    V.collect_vectorized = _spy
    fails = []

    r = run(shared=False)
    print(f"[解耦 master_shared=False] 教师参数量={r['teacher_n']:,}  "
          f"座位对手参数量={r['master_n']:,}  同一对象={r['same_obj']}")
    if r["teacher_n"] != 9_840_000 and r["teacher_n"] < 9_000_000:
        fails.append(f"教师应≈9.84M（d3_it0020），实得 {r['teacher_n']}")
    if r["master_n"] != 767_229:
        fails.append(f"座位对手应=767,229（master），实得 {r['master_n']}")
    if r["same_obj"]:
        fails.append("解耦模式下两者仍是同一对象")

    r2 = run(shared=True)
    print(f"[复用 master_shared=True ] 教师参数量={r2['teacher_n']:,}  "
          f"座位对手参数量={r2['master_n']:,}  同一对象={r2['same_obj']}")
    if not r2["same_obj"]:
        fails.append("shared 模式下应复用同一实例（省一份前向内存）")

    # 教师关闭（β=0）但座位仍要能用：worker 必须能独自建出 master
    main, main_v, nc, main_pk = build("_ckpts/dushan_m2_c5_it3120")
    mas_m, mas_v, _, _ = build("dushan_master")
    V._TEACH_CACHE["model"] = None
    V._MASTER_CACHE["model"] = None
    REC.clear()
    V.collect_parallel(
        main.state_dict(), nc, 8, 1, 1.0, 8.0, 0.05, workers=1,
        past_prob=0.0, snapshots_sd=[], rules="dushan", want_traj=False,
        encoder_version=main_v, encoder_packs=main_pk, snap_key=None,
        explore_eps=0.0, teacher_ck=None, teacher_ver=None, master_prob=1.0,
        master_ck=ck_from_model(mas_m, encoder_version=mas_v),
        master_ver=mas_v, master_shared=True)
    print(f"[教师关闭 β=0]            座位对手参数量={REC.get('master_n')}  "
          f"（旧实现在这里会变成 None ⇒ 一等对手静默消失）")
    if REC.get("master_n") != 767_229:
        fails.append("β=0 时座位对手丢失（旧缺陷复现）")

    V.collect_vectorized = _orig
    if fails:
        print("\n[FAIL] " + "\n       ".join(fails))
        return 1
    print("\n[PASS] klt-teacher 解耦正确：教师与座位对手各走各的缓存，"
          "shared 模式仍复用同一实例，β=0 时座位不丢")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
