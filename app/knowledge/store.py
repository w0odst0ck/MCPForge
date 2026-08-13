"""SQLite 会话与问答日志存储 — data/chat.db（WAL 模式）。

两张表：
- chat_sessions：多轮对话历史（history 列 JSON 存最近 N 轮消息，轮 = user+assistant 两条）
- chat_logs：所有问答端点日志（answer_question / conversation / order_help，
  支撑运营决策：高频问题 Top10、转人工率、命中率）

设计要点：
- 短连接模式：每次操作独立 connect/close，天然线程安全（FastAPI 线程池并发无锁）
- 维护：maintenance() 清理过期会话（TTL 24h）与超保留期日志（30 天），
  首次调用（≈启动）执行一次，之后每日一次（进程内时间戳节流）
- 隐私（沙特 PDPL）：query 含连续 8+ 数字视为疑似手机号等 PII，
  落库以 [PII 已脱敏] 占位，绝不存手机号/地址/姓名
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from app.knowledge.config import kb_settings
from app.utils.log import log

# ── 常量 ──────────────────────────────────────────────────────
# PII 脱敏占位（沙特 PDPL：问答日志不存个人可识别信息）
PII_MASKED = "[PII 已脱敏]"
# 连续 8+ 数字（疑似手机号/身份证号/银行卡号）
_PII_RE = re.compile(r"\d{8,}")

_SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    lang        TEXT,
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    updated_at  TEXT,
    history     TEXT       -- JSON: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
)
"""

_SCHEMA_LOGS = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,          -- conversation 有；单轮为空
    lang        TEXT,
    query       TEXT,
    answer      TEXT,
    hit_count   INTEGER,
    elapsed_ms  INTEGER,
    sources     TEXT,          -- JSON: [doc_id,...]
    is_handoff  INTEGER,       -- 1=转人工（hit_count=0 或 error）
    created_at  TEXT DEFAULT (datetime('now','localtime'))
)
"""


def sanitize_query(query: str) -> str:
    """PII 脱敏：query 含连续 8+ 数字（疑似手机号）时以 [PII 已脱敏] 落库。"""
    if not query:
        return query
    return PII_MASKED if _PII_RE.search(query) else query


class ChatStore:
    """chat.db 的会话与日志存取（短连接模式，线程安全）。"""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = str(db_path or kb_settings.CHAT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # 每日维护节流时间戳（monotonic 秒），None = 尚未执行
        self._last_maintenance: Optional[float] = None

    # ── 连接 ──────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        """新建短连接：WAL 模式 + busy_timeout，读写并发不阻塞太久。"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _execute(self, sql: str, params: tuple = ()) -> int:
        """执行单条写语句并提交，返回影响行数。"""
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(sql, params)
            return cur.rowcount
        finally:
            conn.close()

    def _query_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    # ── 建表 ──────────────────────────────────────────────────
    def init_db(self) -> None:
        """建表（幂等）：chat_sessions + chat_logs + 日志时间索引。"""
        conn = self._connect()
        try:
            with conn:
                conn.execute(_SCHEMA_SESSIONS)
                conn.execute(_SCHEMA_LOGS)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_logs_created ON chat_logs(created_at)")
        finally:
            conn.close()

    # ── 会话 CRUD ─────────────────────────────────────────────
    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """按 session_id 读会话；不存在返回 None。history 反序列化为 list。"""
        row = self._query_one(
            "SELECT session_id, lang, history FROM chat_sessions WHERE session_id = ?", (session_id,)
        )
        if row is None:
            return None
        row["history"] = json.loads(row["history"] or "[]")
        return row

    def upsert_session(self, session_id: str, lang: str, history: list[dict]) -> None:
        """写入/更新会话：首次出现创建，后续续接并刷新 updated_at。"""
        payload = json.dumps(history, ensure_ascii=False)
        self._execute(
            """
            INSERT INTO chat_sessions (session_id, lang, created_at, updated_at, history)
            VALUES (?, ?, datetime('now','localtime'), datetime('now','localtime'), ?)
            ON CONFLICT(session_id) DO UPDATE SET
                lang = excluded.lang,
                updated_at = datetime('now','localtime'),
                history = excluded.history
            """,
            (session_id, lang, payload),
        )

    def delete_expired_sessions(self, ttl_hours: int = 24) -> int:
        """删除 updated_at 早于 now - ttl_hours 的会话，返回删除条数。"""
        return self._execute(
            "DELETE FROM chat_sessions WHERE updated_at < datetime('now','localtime', ?)",
            (f"-{ttl_hours} hours",),
        )

    # ── 问答日志 ──────────────────────────────────────────────
    def append_log(
        self,
        session_id: Optional[str],
        lang: str,
        query: str,
        answer: str,
        hit_count: int = 0,
        elapsed_ms: int = 0,
        sources: Optional[list[str]] = None,
        is_handoff: int = 0,
    ) -> None:
        """插入一条问答日志。query 落库前做 PII 脱敏，sources 序列化为 JSON。"""
        self._execute(
            """
            INSERT INTO chat_logs (session_id, lang, query, answer, hit_count, elapsed_ms, sources, is_handoff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                lang,
                sanitize_query(query),
                answer,
                int(hit_count or 0),
                int(elapsed_ms or 0),
                json.dumps(sources or [], ensure_ascii=False),
                1 if is_handoff else 0,
            ),
        )

    def delete_expired_logs(self, retention_days: int = 30) -> int:
        """删除 created_at 早于 now - retention_days 的日志，返回删除条数。"""
        return self._execute(
            "DELETE FROM chat_logs WHERE created_at < datetime('now','localtime', ?)",
            (f"-{retention_days} days",),
        )

    # ── 运营查询（可选 CLI 或直接 SQL；此处提供便捷函数） ────
    def top_questions(self, limit: int = 10) -> list[dict[str, Any]]:
        """高频问题 TopN：SELECT query, COUNT(*) GROUP BY query ORDER BY COUNT(*) DESC。"""
        return self._query_all(
            "SELECT query, COUNT(*) AS cnt FROM chat_logs GROUP BY query ORDER BY cnt DESC LIMIT ?",
            (int(limit),),
        )

    def handoff_rate(self) -> Optional[float]:
        """转人工率：is_handoff=1 条数 / 总条数；无日志返回 None。"""
        row = self._query_one(
            "SELECT COUNT(*) FILTER (WHERE is_handoff = 1) * 1.0 / COUNT(*) AS rate FROM chat_logs"
        )
        return float(row["rate"]) if row and row["rate"] is not None else None

    # ── 维护 ──────────────────────────────────────────────────
    def maintenance(self) -> None:
        """清理过期会话与超保留期日志。首次调用（≈启动）必执行，之后每日一次。"""
        now = time.monotonic()
        if self._last_maintenance is not None and now - self._last_maintenance < 86400:
            return
        try:
            deleted_sessions = self.delete_expired_sessions(kb_settings.SESSION_TTL_HOURS)
            deleted_logs = self.delete_expired_logs(kb_settings.LOG_RETENTION_DAYS)
            if deleted_sessions or deleted_logs:
                log.info(
                    "chat store maintenance | expired_sessions={} expired_logs={}",
                    deleted_sessions, deleted_logs,
                )
        finally:
            self._last_maintenance = now


# ── 模块级单例（懒加载：首次调用才连库/建表/维护） ─────────────
_chat_store: Optional[ChatStore] = None
_chat_store_lock = threading.Lock()


def get_chat_store() -> ChatStore:
    """返回全局 ChatStore 单例（首次调用初始化：建表 + 启动清理）。"""
    global _chat_store
    if _chat_store is None:
        with _chat_store_lock:
            if _chat_store is None:
                store = ChatStore()
                store.init_db()
                store.maintenance()
                _chat_store = store
    return _chat_store
