"""会话级存储底座 —— 一个会话一个库，进入先清，超时自动清，关闭即清（等价）。

设计动机（2026-10-12 拍板）:
  原方案用全局持久库，"越用越聪明"。改为会话级的原因有三:
  1. 天然隔离: 不同会话不可能互相串数据（跨会话读旧论文结论 = 幻觉固化）;
  2. 无并发写冲突: 不需要锁;
  3. 无隐私残留: 论文与代码的副本只活一个会话。

  代价: 跨论文经验库不再累积（决策 2 降级为"同一会话内复用"）。
        将来要升级，只需把库路径从"会话目录"换成"持久目录"，
        上层代码不动。

三条触发线:
  T1 新会话进入 -> 先清（全新环境）
  T2 同会话再跑 -> 按输入指纹（论文 sha1 + 代码 sha1）查会话索引，
     命中即复用（跳过解析与建索引），未命中才重新解析
  T3 会话超时（TTL）-> 清扫该会话的全部库文件与临时目录
     （兜住"浏览器关闭事件丢失"的情况）

对外口径:
  "关页面即清"在技术上做不到 100% 可靠（浏览器关闭事件不保证到达服务器），
  因此本模块用 T1 + T3 实现等价效果: 下次进入必然全新，关闭后过一个 TTL
  周期自动清干净。两者都实测，不依赖任何浏览器行为。

一条硬边界（与记忆铁律同源）:
  会话记忆里只存**线索**（原文里的值 + 出处 + 整句原文），不存**结论**。
  轨迹账本只记"谁在什么时候用了什么模型、闸门放没放行"，供人复核，
  不供后续步骤当证据用。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from . import synonyms

HERE = Path(__file__).resolve().parent.parent
CACHE_DIR = HERE / "cache"
SESSION_DIR = CACHE_DIR / "sessions"

DEFAULT_TTL = int(os.environ.get("PAPER2LAB_SESSION_TTL") or "3600")        # 秒
SWEEP_INTERVAL = int(os.environ.get("PAPER2LAB_SWEEP_INTERVAL") or "300")   # 秒
_SCHEMA_VERSION = 1

# ---- 状态机（会话索引里一条输入的生命周期）
STATUS_FRESH = "fresh"      # 已上传，尚未解析/建索引
STATUS_INDEXED = "indexed"  # 已解析、已建 FTS5 索引（可复用）
STATUS_VERIFIED = "verified" # 已完成审计（只标记，不触发复用）

# ---------------------------------------------------------------- DDL

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS session_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_index (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    status     TEXT NOT NULL DEFAULT 'fresh',
    created_at REAL    NOT NULL,
    last_used  REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS session_memory (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    idx_id     INTEGER NOT NULL REFERENCES session_index(id),
    param      TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    channel    TEXT    NOT NULL,  -- 线索来自哪一侧: "paper" | "code"
    source     TEXT    NOT NULL,  -- 出处: "Page 6 · §3.1" 或 "train.py:42"
    doc        TEXT    NOT NULL,  -- 原文片段
    created_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS session_trajectory (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    idx_id     INTEGER,
    step       TEXT,
    model      TEXT,
    endpoint   TEXT,
    gate       TEXT,
    elapsed    REAL,
    created_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS fts5_doc (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    idx_id     INTEGER NOT NULL REFERENCES session_index(id),
    doc_id     TEXT    NOT NULL,
    section    TEXT,
    content    TEXT    NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts5_idx USING fts5(
    content, section, tokenize='porter'
);
CREATE INDEX IF NOT EXISTS si_fingerprint ON session_index(fingerprint);
CREATE INDEX IF NOT EXISTS sm_idx ON session_memory(idx_id);
CREATE INDEX IF NOT EXISTS st_idx ON session_trajectory(idx_id);
CREATE INDEX IF NOT EXISTS fd_idx ON fts5_doc(idx_id);
"""

# 建库时顺手把版本号写进 meta，避免旧库被当成新会话误用
_INIT_META = {
    "schema_version": str(_SCHEMA_VERSION),
    "created_at": f"{time.time():.3f}",
}

# ---------------------------------------------------------------- 会话标识

def new_id() -> str:
    """生成一个不可猜、不碰撞的会话 id（不依赖 streamlit，纯函数）。"""
    return f"session-{uuid.uuid4().hex[:12]}"


def fingerprint(pdf_bytes: bytes, code_bytes: bytes = b"") -> str:
    """输入指纹 —— 口径与 app.py 的 meta.fingerprint 严格一致。

    10 位 sha1 截断，够用（一个会话内极少超过 10^5 次输入）。
    """
    return hashlib.sha1(pdf_bytes + code_bytes).hexdigest()[:10]


# ---------------------------------------------------------------- 单例管理器

_stores: dict[str, "SessionStore"] = {}
_current_id: str = ""
_sweeper: Optional[threading.Thread] = None
_sweep_stop = threading.Event()

# 进程级快照缓存: 同会话内复用解析产物（paper/code 对象引用）。
# 不走 JSON 存入 SQL，因为 Evidence 嵌套 + set 字段无法无损序列化。
# T1 清库 + store.clear() 一起清空，保证与库内数据同步。
_snapshot_cache: dict[str, dict] = {}


def set_current_id(sid: str) -> None:
    """由调用方（app.py）显式绑定当前会话 id。"""
    global _current_id
    _current_id = sid or ""


def current_id() -> str:
    """当前会话 id；未绑定时返回空串（表示"无会话，存储不可用"）。"""
    return _current_id


def _start_sweeper() -> None:
    global _sweeper
    if _sweeper is not None and _sweeper.is_alive():
        return
    _sweep_stop.clear()

    def _loop() -> None:
        while not _sweep_stop.is_set():
            try:
                sweep_all()
            except Exception:      # noqa: BLE001
                pass               # 清扫是后台守护，不能把主流程拖垮
            _sweep_stop.wait(SWEEP_INTERVAL)

    _sweeper = threading.Thread(target=_loop, name="session-sweeper", daemon=True)
    _sweeper.start()


def get(*, ttl: int = DEFAULT_TTL, lazy: bool = True) -> "SessionStore":
    """取当前会话的存储（无会话时返回 None，调用方降级为"无缓存"路径）。

    lazy=True（默认）: 首次访问才建库并启动清扫线程。
    """
    sid = current_id()
    if not sid:
        return None
    store = _stores.get(sid)
    if store is None:
        store = SessionStore(SESSION_DIR / f"{sid}.db", ttl=ttl)
        _stores[sid] = store
        if lazy:
            store.init()
        _start_sweeper()
    return store


def close(sid: str) -> Optional["SessionStore"]:
    """关闭并遗忘某个会话的存储（不删文件，交由清扫线程处理）。"""
    store = _stores.pop(sid, None)
    if store is not None:
        store.close()
    return store


def sweep_all(older_than: Optional[float] = None) -> dict:
    """清扫所有超时会话的库文件与临时目录，返回 {"swept": [...], "kept": [...]}。"""
    if older_than is None:
        older_than = time.time() - DEFAULT_TTL
    swept, kept = [], []
    if not SESSION_DIR.exists():
        return {"swept": swept, "kept": kept}
    for db in SESSION_DIR.iterdir():
        if db.suffix != ".db":
            continue
        if _is_expired(db, older_than):
            tmp = SESSION_DIR / db.stem
            _remove_quietly(db)
            _remove_quietly(tmp)
            swept.append(db.name)
            _stores.pop(db.stem, None)
        else:
            kept.append(db.name)
    return {"swept": swept, "kept": kept}


def _is_expired(db: Path, cutoff: float) -> bool:
    """优先读库内 session_meta.created_at；库打不开时降级为文件 mtime。"""
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT value FROM session_meta WHERE key='created_at'"
            ).fetchall()
        finally:
            conn.close()
        if rows:
            return float(rows[0][0]) < cutoff
    except Exception:      # noqa: BLE001
        pass
    try:
        return db.stat().st_mtime < cutoff
    except OSError:
        return False


def _remove_quietly(p: Path) -> None:
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------- 存储对象

class SessionStore:
    """一个会话一份 SQLite（含 FTS5），生命周期由 T1/T3 控制。

    调用约定:
      store = session.get()
      if store is None:  # 无会话 -> 走无缓存路径
          ...
    """

    def __init__(self, path: Path, *, ttl: int = DEFAULT_TTL):
        self.path = path
        self.ttl = ttl
        self.sid = path.stem
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._endpoint_cache: dict = {}

    # ---- 连接

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.path))
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # ---- T1: 进入先清

    def init(self) -> dict:
        """创建/重建本会话的库（T1 调用）。返回可展示的回执。

        库已存在 -> 删掉重来（保证"全新环境"，旧线索不再污染）。
        """
        if self.path.exists():
            self.close()
            _remove_quietly(self.path)
            # WAL 模式会多留 -shm / -wal 陪葬文件
            for suf in ("-shm", "-wal"):
                _remove_quietly(self.path.parent / (self.path.name + suf))
        _snapshot_cache.pop(self.sid, None)
        conn = self._connect()
        conn.executescript(_CREATE_TABLES)
        conn.executemany(
            "INSERT OR REPLACE INTO session_meta(key, value) VALUES (?, ?)",
            [(k, v) for k, v in _INIT_META.items()],
        )
        conn.commit()
        return {"sid": self.sid, "status": "initialized", "path": str(self.path)}

    def clear(self) -> dict:
        """T1 的"清"：删库 + 清进程快照 + 删本会话的临时目录。
        init() 内部已调用，一般不需要单独调（供外部清扫脚本用）。
        """
        self.close()
        _remove_quietly(self.path)
        for suf in ("-shm", "-wal"):
            _remove_quietly(self.path.parent / (self.path.name + suf))
        _remove_quietly(SESSION_DIR / self.sid)
        _snapshot_cache.pop(self.sid, None)
        return {"sid": self.sid, "status": "cleared"}

    # ---- T2: 指纹复用

    def lookup(self, fp: str) -> Optional[str]:
        """查输入指纹的状态；未建索引返回 None（调用方应重新解析）。"""
        if not fp:
            return None
        conn = self._connect()
        rows = conn.execute(
            "SELECT status, id FROM session_index WHERE fingerprint=? LIMIT 1",
            (fp,),
        ).fetchone()
        if not rows:
            return None
        status, idx_id = rows
        conn.execute(
            "UPDATE session_index SET last_used=? WHERE id=?",
            (time.time(), idx_id),
        )
        conn.commit()
        return status

    def mark(self, fp: str, status: str) -> Optional[int]:
        """记录一个输入已到达某状态，返回其 session_index.id。"""
        conn = self._connect()
        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO session_index(fingerprint, status, created_at, last_used)"
            " VALUES(?, ?, ?, ?)",
            (fp, status, now, now),
        )
        conn.execute(
            "UPDATE session_index SET status=?, last_used=? WHERE fingerprint=?",
            (status, now, fp),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT id FROM session_index WHERE fingerprint=? LIMIT 1", (fp,)
        ).fetchone()
        return rows[0] if rows else None

    # ---- 会话记忆（只记线索）

    def store_memory(
        self, fp: str, param: str, value: str, channel: str, source: str, doc: str
    ) -> int:
        """记一条线索（无出处不许入库，由调用方保证 source/doc 非空）。

        channel 只能是 "paper" 或 "code"，告诉装配器这条线索来自哪一侧。
        """
        if channel not in ("paper", "code"):
            raise ValueError(f"channel 必须是 'paper' 或 'code'，给了 {channel!r}")
        idx_id = self.mark(fp, STATUS_INDEXED)
        conn = self._connect()
        cur = conn.execute(
            "INSERT INTO session_memory(idx_id, param, value, channel, source, doc,"
            " created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (idx_id, param, value, channel, source, doc, time.time()),
        )
        conn.commit()
        return cur.lastrowid

    def recall(self, fp: str) -> list[dict]:
        """取某篇论文的全部线索（空列表 = 未见过）。

        每条线索带 channel（"paper" | "code"），装配器按它把论文侧与代码侧分开。
        """
        idx_id = self._idx_id(fp)
        if idx_id is None:
            return []
        conn = self._connect()
        rows = conn.execute(
            "SELECT param, value, channel, source, doc FROM session_memory"
            " WHERE idx_id=? ORDER BY id",
            (idx_id,),
        ).fetchall()
        return [
            {"param": r[0], "value": r[1], "channel": r[2],
             "source": r[3], "doc": r[4]}
            for r in rows
        ]

    # ---- 轨迹账本（只写不读）

    def record(
        self,
        fp: str,
        step: str,
        *,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        gate: Optional[str] = None,
        elapsed: float = 0.0,
    ) -> int:
        """记一个动作。model/endpoint 不传则自动解析（与 resolve_endpoint 同源）。"""
        idx_id = self.mark(fp, STATUS_INDEXED)
        if model is None:
            model, endpoint = self._resolve_once()
        conn = self._connect()
        cur = conn.execute(
            "INSERT INTO session_trajectory(idx_id, step, model, endpoint, gate,"
            " elapsed, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (idx_id, step, model, endpoint, gate, elapsed, time.time()),
        )
        conn.commit()
        return cur.lastrowid

    def trajectory(self, fp: str) -> list[dict]:
        """取某篇论文的轨迹（给人看/给审计用，正常流程不读它）。"""
        idx_id = self._idx_id(fp)
        if idx_id is None:
            return []
        conn = self._connect()
        rows = conn.execute(
            "SELECT step, model, endpoint, gate, elapsed, created_at FROM"
            " session_trajectory WHERE idx_id=? ORDER BY id",
            (idx_id,),
        ).fetchall()
        return [
            {
                "step": r[0], "model": r[1], "endpoint": r[2], "gate": r[3],
                "elapsed": r[4], "at": r[5],
            }
            for r in rows
        ]

    def _idx_id(self, fp: str) -> Optional[int]:
        if not fp:
            return None
        rows = self._connect().execute(
            "SELECT id FROM session_index WHERE fingerprint=? LIMIT 1", (fp,)
        ).fetchone()
        return rows[0] if rows else None

    def _resolve_once(self) -> tuple[str, str]:
        """一次会话内只解析一次端点（避免重复导入/重复查 env）。"""
        cache = self._endpoint_cache
        if "cached" in cache:
            return cache["model"], cache["endpoint"]
        try:
            from .llm import resolve_endpoint

            _k, endpoint, model = resolve_endpoint()  # 返回 (key, url, model)
        except Exception:      # noqa: BLE001
            model, endpoint = "", ""
        cache["model"], cache["endpoint"] = model, endpoint
        cache["cached"] = True
        return model, endpoint

    # ---- 进程快照（快路径：复用本轮解析产物，不重复解析 PDF/代码）

    def store_snapshot(self, fp: str, paper: object, code: object) -> None:
        """把本轮解析产物记入进程快照（同会话内复用）。"""
        self.mark(fp, STATUS_INDEXED)
        _snapshot_cache.setdefault(self.sid, {})[fp] = (paper, code)

    def load_snapshot(self, fp: str) -> Optional[tuple]:
        """取快照；未存过返回 None（调用方应重新解析）。"""
        return _snapshot_cache.get(self.sid, {}).get(fp)

    # ---- FTS5 全文索引（RAG 的检索层）

    def index_doc(self, fp: str, doc_id: str, section: str, content: str) -> None:
        """把一个文档段送入索引（可多次调用，幂等追加）。"""
        idx_id = self.mark(fp, STATUS_INDEXED)
        conn = self._connect()
        conn.execute(
            "INSERT INTO fts5_doc(idx_id, doc_id, section, content) VALUES(?, ?, ?, ?)",
            (idx_id, doc_id, section, content),
        )
        conn.execute(
            "INSERT INTO fts5_idx(rowid, content, section) VALUES(NULL, ?, ?)",
            (content, section),
        )
        conn.commit()

    def search(
        self, query: str, *, limit: int = 20, synonyms: bool = True
    ) -> list[dict]:
        """全文检索，返回原文段落 + 出处。同义词默认展开。

        示例: query="learning rate" -> 实际搜 "learning rate OR lr OR 学习率"
        """
        expanded = self._expand(query) if synonyms else query
        conn = self._connect()
        rows = conn.execute(
            "SELECT rowid, section, highlight(fts5_idx, 0, '「', '」')"
            " FROM fts5_idx WHERE fts5_idx MATCH ? ORDER BY rank LIMIT ?",
            (expanded, limit),
        ).fetchall()
        return [
            {"rowid": r[0], "section": r[1], "snippet": self._clip(r[2])}
            for r in rows
        ]

    @staticmethod
    def _clip(text: str, limit: int = 200) -> str:
        """把 highlight 的高亮全文截成一段（≤limit 字符），保留高亮标记。"""
        if len(text) <= limit:
            return text
        # 以词为单位截断，避免在汉字中间或英文单词中间切开
        i = limit
        while i > 0 and not text[i].isspace() and text[i] not in (
            "，", "。", "、", ";", "；", "）", ")"
        ):
            i -= 1
        if i == 0:
            i = limit
        return text[:i].rstrip() + "…"

    @staticmethod
    def _expand(term: str) -> str:
        """把参数名展开成 "自身 OR 同义词"（FTS5 MATCH 语法）。"""
        key = synonyms.canonical_key(term)
        if not key:
            return f'"{term}"'
        disp, aliases = synonyms.PARAM_SYNONYMS.get(key, ("", []))
        terms = [term] + [a for a in aliases if a != term]
        # 同义词表里中英文混排，全部按 FTS5 的词形索引，直接 OR 即可
        return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))

    # ---- 快照/统计

    def info(self) -> dict:
        conn = self._connect()
        n_idx = conn.execute("SELECT COUNT(*) FROM session_index").fetchone()[0]
        n_mem = conn.execute("SELECT COUNT(*) FROM session_memory").fetchone()[0]
        n_traj = conn.execute("SELECT COUNT(*) FROM session_trajectory").fetchone()[0]
        n_doc = conn.execute("SELECT COUNT(*) FROM fts5_doc").fetchone()[0]
        return {
            "sid": self.sid, "path": str(self.path),
            "inputs": n_idx, "memories": n_mem, "trajectories": n_traj,
            "indexed_docs": n_doc, "ttl": self.ttl,
        }


def cleanup() -> dict:
    """一次性清扫全部超时会话（CLI / 定时任务入口）。"""
    return sweep_all()