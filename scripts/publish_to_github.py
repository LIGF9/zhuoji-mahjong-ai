"""一键把本地仓库发布到 GitHub（建仓 + 首次推送）。

只需一个 Personal Access Token，其余（用户名、邮箱、仓库地址）全部自动获取。
不依赖 gh CLI，只用标准库 + git。

用法::

    # 1) 把 token 存到文件（不要放进命令行参数，避免进入 shell 历史）
    #    文件内容就是一行 token，例如 ghp_xxxxxxxx
    python scripts/publish_to_github.py --token-file .gh_token \
        --repo zhuoji-mahjong-ai --public \
        --message-file .commit_msg

    # 2) 只想检查 token 和本地状态，不真推：
    python scripts/publish_to_github.py --token-file .gh_token --dry-run

安全说明
--------
* token 只从文件或 ``GITHUB_TOKEN`` 环境变量读取，**不出现在命令行参数里**；
* 推送用一次性 URL，**不会把 token 写进 ``.git/config``**（推送后 remote 被重置为
  不含 token 的干净地址）；
* 脚本只把 token 用于 GitHub API 与本次 push，不落盘、不回显。

Token 需要的最小权限（classic）：``repo``；若只想建公开仓库，``public_repo`` 即可。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(ROOT), text=True,
                          capture_output=True, **kw)


def api(path: str, token: str, method: str = "GET", payload: dict | None = None):
    """调用 GitHub REST API。"""
    url = path if path.startswith("http") else API + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "zhuoji-mahjong-publisher")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            body = json.loads(body)
        except Exception:  # noqa: BLE001
            pass
        return exc.code, body
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}


def read_token(args) -> str:
    if args.token_file:
        p = Path(args.token_file)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            sys.exit(f"找不到 token 文件：{p}")
        return p.read_text(encoding="utf-8").strip()
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not tok:
        sys.exit("未提供 token：用 --token-file 或设置 GITHUB_TOKEN 环境变量。")
    return tok.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--repo", default="zhuoji-mahjong-ai")
    ap.add_argument("--owner", default=None,
                    help="默认推到自己账号下；填组织名可推到组织")
    ap.add_argument("--public", action="store_true", help="建公开仓库（默认私有）")
    ap.add_argument("--description", default=None)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message-file", default=None, help="提交信息文件")
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-commit", action="store_true",
                    help="跳过自动提交（已提交过时用）")
    args = ap.parse_args()

    priv = not args.public
    token = read_token(args)

    # ---- 1. 校验 token，拿用户名 ----------------------------------------
    status, user = api("/user", token)
    if status != 200:
        sys.exit(f"token 校验失败（HTTP {status}）：{user}")
    login = user["login"]
    uid = user["id"]
    email = user.get("email") or f"{uid}+{login}@users.noreply.github.com"
    print(f"[1/5] token 有效：{login}（id={uid}）")

    owner = args.owner or login
    print(f"      目标仓库：{owner}/{args.repo}  "
          f"({'公开' if not priv else '私有'}）")

    if args.dry_run:
        print("\n[dry-run] 仅校验，不建仓、不推送。")
        return 0

    # ---- 2. 配置 git 身份（仓库级，不动全局配置）-------------------------
    for k, v in (("user.name", login), ("user.email", email)):
        if run(["git", "config", "--local", k, v]).returncode != 0:
            sys.exit(f"设置 {k} 失败")
    print(f"[2/5] git 身份：{login} <{email}>（仅本仓库）")

    # ---- 3. 提交 ---------------------------------------------------------
    if not args.skip_commit:
        if args.message_file:
            mf = Path(args.message_file)
            msg = (mf if mf.is_absolute() else ROOT / mf).read_text(encoding="utf-8")
        else:
            msg = args.message or "Initial commit: 幺鸡 · 贵州捉鸡麻将深度学习智能体"
        ignored = run(["git", "add", "-A"])
        if ignored.returncode != 0:
            sys.exit(f"git add 失败：{ignored.stderr}")
        committed = run(["git", "commit", "-F", "-"], input=msg)
        if committed.returncode != 0:
            if "nothing to commit" in (committed.stdout + committed.stderr):
                print("[3/5] 无新改动，跳过提交")
            else:
                sys.exit(f"git commit 失败：\n{committed.stdout}\n{committed.stderr}")
        else:
            print(f"[3/5] 已提交：{msg.splitlines()[0]}")
    else:
        print("[3/5] 跳过提交")

    # ---- 4. 建仓库（已存在则复用）---------------------------------------
    if args.owner and args.owner != login:
        path = f"/orgs/{args.owner}/repos"
    else:
        path = "/user/repos"
    status, repo = api(path, token, "POST", {
        "name": args.repo,
        "private": priv,
        "description": args.description or "幺鸡 · 贵州捉鸡麻将深度学习智能体",
        "has_issues": True, "has_wiki": False, "has_projects": False,
        "auto_init": False,
    })
    if status == 201:
        print(f"[4/5] 已创建仓库 {repo['full_name']}")
    elif status == 422 and "already exists" in str(repo).lower():
        status2, repo = api(f"/repos/{owner}/{args.repo}", token)
        print(f"[4/5] 仓库已存在，复用 {owner}/{args.repo}")
    else:
        sys.exit(f"建仓失败（HTTP {status}）：{repo}")

    clone_url = repo.get("clone_url") or f"https://github.com/{owner}/{args.repo}.git"
    html_url = repo.get("html_url", f"https://github.com/{owner}/{args.repo}")
    default_branch = repo.get("default_branch") or args.branch

    # ---- 5. 推送（token 只出现在本次进程参数里，不写进 .git/config）------
    auth_url = clone_url.replace("https://", f"https://{login}:{token}@")
    run(["git", "remote", "remove", "origin"])
    run(["git", "remote", "add", "origin", clone_url])      # 干净地址

    print(f"[5/5] 正在推送 {args.branch} → {owner}/{args.repo} ...")
    push = run(["git", "push", auth_url, f"{args.branch}:{args.branch}", "--force"])
    if push.returncode != 0:
        # 远端可能已有初始提交，尝试先拉后推
        print("      直推失败，尝试先合并远端历史再推 ...")
        run(["git", "fetch", auth_url, default_branch])
        run(["git", "merge", "--allow-unrelated-histories", "-m",
             "merge remote initial history", "FETCH_HEAD"])
        push = run(["git", "push", auth_url, f"{args.branch}:{args.branch}", "--force"])
    if push.returncode != 0:
        sys.exit(f"推送失败：\n{push.stdout}\n{push.stderr}")

    run(["git", "branch", f"--set-upstream-to=origin/{args.branch}", args.branch])
    print(f"\n完成 → {html_url}")
    print("提示：remote 已重置为不含 token 的地址，token 未被写入 .git/config。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
