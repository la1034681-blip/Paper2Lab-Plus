"""阶段二 · 会话级存储底座验证脚本。

不依赖 pytest, 纯命令行; 输出到屏幕与 cache/session_probe.txt 两份,
方便复查时直接跑。
"""
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from paper2lab import housekeeping as hk, session  # noqa: E402

CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache"
OUT = CACHE / "session_probe.txt"
lines: list[str] = []
fails = []


def check(ok: bool, msg: str) -> None:
    line = ("  OK   " if ok else "  FAIL ") + msg
    lines.append(line)
    print(line)
    if not ok:
        fails.append(msg)


def bar() -> None:
    lines.append("=" * 66)
    print("=" * 66)


bar()
lines.append("阶段二 · 会话级存储底座验证")
print("阶段二 · 会话级存储底座验证")
bar()

# ---------------------------------------------------------------- 0. 环境
print()
lines.append("")
print("0. 环境")
store = session.get()
check(store is None, "未绑定时 session.get() 返回 None(不自动建库)")
print(f"   python: {sys.version.split()[0]} | sqlite: {__import__('sqlite3').sqlite_version}")
lines.append(f"   python: {sys.version.split()[0]} | sqlite: {__import__('sqlite3').sqlite_version}")

# ---------------------------------------------------------------- 1. 会话 id 与绑定
print()
lines.append("")
print("1. 会话 id 与绑定")
sid = session.new_id()
check(len(sid) == 20 and sid.startswith("session-"), f"new_id() 格式正确: {sid}")
session.set_current_id(sid)
check(session.current_id() == sid, f"set_current_id 生效: {session.current_id()}")
s2 = session.get()
check(s2 is not None, "绑定后 get() 返回有效 store")
check(s2.sid == sid, f"store.sid 一致: {s2.sid}")

# ---------------------------------------------------------------- 2. T1 进入先清
print()
lines.append("")
print("2. T1 新会话进入 → 先清")
s2.init()
s2.store_memory(sid, "learning_rate", "1e-4", "paper", "§3.1", "doc1")
check(s2.info()["memories"] == 1, "写入一条线索后 memory=1")
ck = hk.session_clear()
check(ck == sid, f"session_clear 返回当前 sid: {ck}")
check(s2.info()["memories"] == 0, "session_clear 后 memory=0(旧库被清)")
fp = session.fingerprint(b"paper A", b"code A")
rep = hk.session_lookup(b"paper A", b"code A")
check(rep is not None, "session_lookup 返回非 None")
check(rep["fingerprint"] == fp, f"指纹正确: {rep['fingerprint']}")
check(rep["sid"] == sid, f"sid 一致: {rep['sid']}")
check(rep["needs_index"] is True, "新索引 → needs_index=True")
check(
    session.get().info()["memories"] == 0,
    "T1 先清: 旧线索被清掉",
)

# ---------------------------------------------------------------- 3. T2 指纹复用
print()
lines.append("")
print("3. T2 同会话再跑 → 指纹复用")
s3 = session.get()
rep = hk.session_lookup(b"paper A", b"code A")
check(rep["needs_index"] is True, "T1 后首次 → needs_index=True")
s3.index_doc(fp, "paper-01", "§3.1", "learning rate is 1e-4")
check(len(s3.search("learning")) >= 1, "索引后可搜到英文")
rep = hk.session_lookup(b"paper A", b"code A")
check(rep["status"] == "indexed", f"已索引的指纹返回 status=({rep['status']})")
check(rep["needs_index"] is False, "needs_index=False(跳过解析)")
rep2 = hk.session_lookup(b"paper A" + b"x", b"code A")
check(rep2["needs_index"] is True, "指纹变化后 needs_index=True")

# ---------------------------------------------------------------- 4. FTS5 检索 + 同义词
print()
lines.append("")
print("4. FTS5 全文索引 + 同义词展开")
rows = s3.search("learning rate")
check(len(rows) >= 1, f"英文全文检索命中 {len(rows)} 条")
check("「learning rate」" in rows[0]["snippet"], "高亮标记正确")
rows = s3.search("lr")
check(len(rows) >= 1, f"同义词 'lr' 命中 {len(rows)} 条")
rows = s3.search("学习率")
check(len(rows) >= 1, f"中文同义词'学习率'命中 {len(rows)} 条")
rows = s3.search("learning rate", synonyms=False)
check(len(rows) >= 1, f"同义词关闭时原文短语仍命中({len(rows)} 条，纯透传)")
rows = s3.search("lr", synonyms=False)
check(len(rows) == 0, f"同义词关闭后纯别名'lr'不命中({len(rows)} 条)")

# ---------------------------------------------------------------- 5. 会话记忆(只记线索)
print()
lines.append("")
print("5. 会话记忆")
s3.store_memory(fp, "learning_rate", "1e-4", "paper", "§3.1", "The learning rate is 1e-4")
mem = s3.recall(fp)
check(len(mem) >= 1, f"写入后可读出 {len(mem)} 条线索")
check(mem[-1]["value"] == "1e-4", f"值正确: {mem[-1]['value']}")
check(mem[-1]["source"] == "§3.1", f"出处正确: {mem[-1]['source']}")
check("The learning rate" in mem[-1]["doc"], f"原文可核验: {mem[-1]['doc'][:32]}…")
check(s3.recall("no-such-fp") == [], "未见过论文返回空列表")

# ---------------------------------------------------------------- 6. 轨迹账本(模型溯源)
print()
lines.append("")
print("6. 轨迹账本 · 模型溯源列")
s3.record(fp, "parse_paper", elapsed=0.42)
s3.record(fp, "align", elapsed=1.10)
t = s3.trajectory(fp)
check(len(t) == 2, f"记入 2 条轨迹")
# 口径（2026-10-14 修）: 断言"账本记的 = .env 里配的"，**不写死模型名**。
# 主力模型是会换的（书生 intern-s2 -> DeepSeek -> 以后的本地开源模型），
# 写死模型名会让这条测试变成"模型的看门人"而不是"账本口径的看门人"：
# 换模型时它报警，但它其实什么都没坏。
from paper2lab.envfile import get as _env_get
_exp_model = _env_get("LLM_MODEL") or _env_get("DEEPSEEK_MODEL") or "deepseek-chat"
_exp_url = (_env_get("LLM_BASE_URL") or _env_get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com")
for rec in t:
    check(rec["model"] == _exp_model,
          f"账本模型 = .env 配置（{_exp_model}）: {rec['model']}")
    check(rec["endpoint"] == _exp_url,
          f"账本端点 = .env 配置（{_exp_url}）: {rec['endpoint']}")
saved = {k: os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")}
for k in saved:
    os.environ.pop(k, None)
sid4 = session.new_id()
session.set_current_id(sid4)
s4 = session.get()
s4.init()
s4.record(fp + "x", "aux")
t4 = s4.trajectory(fp + "x")
check(t4[0]["model"] == "deepseek-chat", f"无 .env 时回落 deepseek-chat: {t4[0]['model']}")
for k, v in saved.items():
    if v is not None:
        os.environ[k] = v

# ---------------------------------------------------------------- 7. T3 超时清扫
print()
lines.append("")
print("7. T3 会话超时 → TTL 清扫")
conn = s2._connect()
old = time.time() - 70
conn.execute("UPDATE session_meta SET value=? WHERE key=?", (str(old), "created_at"))
conn.commit()
s2.close()
check(session._is_expired(s2.path, time.time() - 30) is True, "is_expired 判过期(30s)")
check(session._is_expired(s2.path, time.time() - 90) is False, "is_expired 判未过期(90s)")

# ---------------------------------------------------------------- 8. A/B 线隔离
print()
lines.append("")
print("8. A 线(force=False) / B 线(force=True 默认) 隔离")
(CACHE / "llm_cache.json").write_text('{"probe": 1}', encoding="utf-8")
r = hk.purge_run_state(force=False)
check(not any(k["kind"] == "运行缓存" for k in r.get("cleared", [])),
      "A 线: 运行缓存未被清")
check(CACHE.joinpath("llm_cache.json").exists(), "A 线: llm_cache.json 仍在")
r2 = hk.purge_run_state()
check(any(k["kind"] == "运行缓存" for k in r2.get("cleared", [])),
      "B 线(默认): 运行缓存被清")
check(hk.DEFAULT_TTL == 3600, f"DEFAULT_TTL={hk.DEFAULT_TTL} 与 session 同源")

# ---------------------------------------------------------------- 9. 降级路径
print()
lines.append("")
print("9. 无会话时降级")
session.set_current_id("")
check(hk.session_lookup(b"paper", b"code") is None, "无绑定返回 None(降级)")
check(hk.session_clear() is None, "无绑定时 session_clear 返回 None")
sid5 = session.new_id()
session.set_current_id(sid5)
hk.session_lookup(b"paper", b"code")

print()
lines.append("")
bar()
if fails:
    lines.append(f"未通过 {len(fails)} 项:")
    print(f"未通过 {len(fails)} 项:")
    for f in fails:
        lines.append("  -" + f)
        print("  -", f)
else:
    lines.append(f"全部通过({os.path.basename(__file__)})")
    print(f"全部通过({os.path.basename(__file__)})")
bar()

OUT.write_text("\n".join(lines), encoding="utf-8")