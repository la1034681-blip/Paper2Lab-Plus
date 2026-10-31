"""多用户隔离验证 —— 会话 id 是**线程局部**的, 一个进程服务多个访问者也不串。

守的契约:
  A 线程局部语义: 一个线程绑定的 id 不影响另一个线程
  B 未绑定的线程 = 无会话: get() 返回 None, **不自动建库, 更不读别人的库**
  C 同一个 sid 跨线程仍是**同一个 store**（sid 才是键, 线程只是"当前是谁"的入口）
  D 不同 sid 的记忆/轨迹互不可见 —— 这是**隐私级**断言, 不是显示问题
  E 线程复用: 同一线程第二次进入若**不重新绑定**, 会读到上次的值（所以入口必须每次绑）
  F 静态形态: 模块里不再有进程级会话 id 全局; session 模块不依赖 streamlit;
    app.py 的绑定位于按钮分支**之前**（每次 rerun 都执行）
  G 端到端: 两个线程各自走一轮 T1 清库 + 写线索, 结果互不污染

为什么要有这一项: 公开链接是**一个进程服务所有访问者**。把"当前是谁"放在进程级
全局变量里, 单机单人跑一万次都看不出问题, 一到多用户就会串 ——
而且串的是别人的论文线索与轨迹账本。这条属于"没坏过但一定会坏"的那类缺陷,
所以必须由**可执行断言**盯着, 不能靠注释保证。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import session  # noqa: E402

CHECKS = 0
FAILURES = []
_BOX: dict = {}


def check(ok, label):
    global CHECKS
    CHECKS += 1
    assert isinstance(ok, bool), "[%s] ok 不是布尔值: %r" % (label, ok)
    assert isinstance(label, str)
    if not ok:
        FAILURES.append(label)
        print("  *x* " + label)
    else:
        print("  [ok] " + label)


def runaway(name: str, fn, *a, **kw):
    """在**新线程**里跑 fn 并取回结果（线程里抛异常不会打断主线程）。"""
    def _wrap():
        try:
            _BOX[name] = ("ok", fn(*a, **kw))
        except BaseException as ex:      # noqa: BLE001
            # 必须 catch BaseException：线程里被 SystemExit / 安全钩子打断时，
            # 只 catch Exception 会让 _BOX 根本没写入，主线程报 KeyError ——
            # 报错信息变成"取不到结果"，真正的原因（谁中断了线程）被吞掉。
            # 实证 2026-10-29：一次批量删除被安全策略拦下（SystemExit），
            # 测试报的是 KeyError: 'g1'，看不出是被拦了。
            _BOX[name] = ("err", ex)

    t = threading.Thread(target=_wrap, name="probe-" + name)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "线程 %s 超时未结束（可能死锁）" % name
    kind, val = _BOX.pop(name)
    if kind == "err":
        raise val
    return val


sid_a, sid_b = session.new_id(), session.new_id()

# ---------------------------------------------------------------- A. 线程局部语义
print("\n== A. 线程局部语义 ==")
session.set_current_id(sid_a)
check(session.current_id() == sid_a, "主线程绑定 A 后读到 A: %s" % session.current_id())
check(runaway("a1", session.current_id) == "",
      "**新线程未绑定 → 读到空串**（不是主线程的 A）")


def _bind_read(sid):
    session.set_current_id(sid)
    return session.current_id()


check(runaway("a2", _bind_read, sid_b) == sid_b, "另一线程绑 B 后读到 B")
check(session.current_id() == sid_a, "主线程仍读到 A（互不影响）")

# ---------------------------------------------------------------- B. 未绑定 = 无会话
print("\n== B. 未绑定线程 = 无会话（降级, 绝不读别人的库）==")
check(runaway("b1", session.get) is None, "未绑定线程 get() 返回 None")
check(runaway("b2", session.current_id) == "", "未绑定线程 current_id() 为空")


# ---------------------------------------------------------------- C. 同 sid 跨线程同一 store
def _get_store(sid):
    session.set_current_id(sid)
    return session.get()


print("\n== C. 同一个 sid 跨线程 → 同一个 store ==")
s_a = _get_store(sid_a)
s_a_thread = runaway("c1", _get_store, sid_a)
check(s_a_thread is s_a, "同 sid 在两个线程取到同一个 store 对象")
check(s_a_thread.sid == sid_a, "store.sid 与绑定一致: %s" % s_a_thread.sid)
check(runaway("c2", _get_store, sid_b) is not s_a, "不同 sid 是不同 store")


# ---------------------------------------------------------------- D. 记忆互不可见
print("\n== D. 不同 sid 的记忆互不可见（隐私级）==")
s_a.init()
s_b = runaway("d1", _get_store, sid_b)
s_b.init()
FP = "fp-shared-shape"
s_a.store_memory(FP, "learning_rate", "1e-4", "paper", "Page 6 · §3.1", "论文原文 A")
check(len(s_a.recall(FP)) == 1, "A 会话写入 1 条线索")
check(s_b.recall(FP) == [], "**B 会话读不到 A 的线索**")
s_b.store_memory(FP, "batch_size", "64", "code", "train.py:1", "代码原文 B")
check([m["param"] for m in s_b.recall(FP)] == ["batch_size"],
      "B 会话只有自己的线索: %s" % [m["param"] for m in s_b.recall(FP)])
check([m["param"] for m in s_a.recall(FP)] == ["learning_rate"],
      "A 会话仍只有自己的线索（没被 B 覆盖）")
# 轨迹账本同理: 只记自己的动作
s_a.record(FP, "parse_paper")
check(len(s_a.trajectory(FP)) == 1 and s_b.trajectory(FP) == [],
      "轨迹账本同样各记各的")

# ---------------------------------------------------------------- E. 线程复用不残留
print("\n== E. 线程复用：入口必须每次重新绑定 ==")
check(runaway("e1", _bind_read, sid_b) == sid_b, "复用的线程重新绑 B → 读到 B")
check(session.current_id() == sid_a, "主线程不受该线程影响")
check(runaway("e2", _get_store, sid_a) is s_a, "线程复用时同 sid 仍拿到同一个 store")

# ---------------------------------------------------------------- F. 静态形态
print("\n== F. 静态形态（防止改动把隔离退回全局变量）==")
_src = (ROOT / "paper2lab" / "session.py").read_text(encoding="utf-8")
check("threading.local()" in _src, "会话 id 用 threading.local（不是进程级全局）")
# 注意别写成 `"_current_id" in _src` —— 函数名 set_current_id 就含这个子串,
# 那样写永远不通过（这个断言自己踩过一次）。要盯的是**全局变量的定义与改写**。
check("global _current_id" not in _src and "_current_id: str" not in _src,
      "旧的进程级全局 _current_id 已彻底移除（定义与 global 改写都没有）")
check("_current.sid" in _src, "读写都走线程局部 _current.sid")
check("import streamlit" not in _src, "session 模块不依赖 streamlit（CLI/MCP 同用）")
_app = (ROOT / "app.py").read_text(encoding="utf-8")
check("_set_current_id(st.session_state[" in _app,
      "app.py 把会话 id 绑到当前线程")
_top = _app.index("_set_current_id(st.session_state[")
_btn = _app.index("\nif run:")
check(_top < _btn, "绑定位于按钮分支**之前**（每次 rerun 都执行，不只点按钮那次）")
check(_app.count("set_current_id(") >= 2, "run 分支里还有一次绑定（首次签发时补绑）")

# ---------------------------------------------------------------- G. 端到端双访问者
print("\n== G. 端到端：两个线程各跑一轮（模拟两个访问者）==")


def _visitor(sid, param, value):
    session.set_current_id(sid)
    store = session.get()
    store.init()                     # T1 进入先清
    store.store_memory(FP, param, value, "paper", "Page 1", "doc")
    return [m["param"] for m in store.recall(FP)]


r1 = runaway("g1", _visitor, sid_a, "seed", "42")
r2 = runaway("g2", _visitor, sid_b, "epochs", "10")
check(r1 == ["seed"], "访问者 A 这一轮只看到自己的线索: %s" % r1)
check(r2 == ["epochs"], "访问者 B 这一轮只看到自己的线索: %s" % r2)
check(len(runaway("g3", _visitor, sid_a, "momentum", "0.9")) == 1,
      "A 第二轮 T1 清库后只剩本轮线索（没有跨轮残留）")
check(runaway("g4", _visitor, sid_b, "weight_decay", "1e-2") == ["weight_decay"],
      "A 的第二轮没有影响 B")

# ---------------------------------------------------------------- 清理 + 汇总
for _sid in (sid_a, sid_b):
    _s = session._stores.pop(_sid, None)
    if _s is not None:
        _s.clear()

print("\n%d 项, 失败 %d" % (CHECKS, len(FAILURES)))
if FAILURES:
    for f in FAILURES:
        print("  - " + f)
sys.exit(1 if FAILURES else 0)
