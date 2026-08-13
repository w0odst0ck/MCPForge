"""Mojin AI 客服全链路健康检查 — 供日常巡检 / cron 使用。

用法（在 MCPForge 目录下）:
    uv run python scripts/healthcheck.py            # 人类可读输出
    uv run python scripts/healthcheck.py --json     # 纯 JSON（cron/脚本用，key 固定）

退出码: 0 = 全绿（或仅 ⚠️ 可恢复告警）；1 = 存在 ❌ 致命项。

检查项:
    1. model server:   GET /healthz，确认 status=ok 且 models 含 bge-m3 / bge-reranker-v2-m3
    2. Milvus Lite:    打开 MILVUS_URI 对应 .db，list_collections 正常
    3. 集合完整性:      6 个集合（kb/products × en/cn/ar）都存在 + 实体数统计
                       （kb >= 8 块 / 语言，products == 10 块 / 语言；数量偏差记 ⚠️）
    4. DeepSeek key:   DEEPSEEK_API_KEY 是否存在（不打印值）+ 极小调用验证（max_tokens=1）
    5. 工具链路冒烟:   search_products / answer_question(转人工兜底) / order_help 结构校验

安全约束: 不打印 DEEPSEEK_API_KEY 的值；Milvus 被 uvicorn 占用时记 ⚠️ 而非崩溃。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# 抑制 pkg_resources 弃用警告（jieba / milvus-lite 的 import 会触发），保证 --json 输出纯净
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# 保证从任意 cwd 都能 import app 包（uv run 在 MCPForge 目录下执行）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger as _loguru  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402
from rag_toolkit.storage.milvus_manager import MilvusManager  # noqa: E402

from app.knowledge.config import kb_settings  # noqa: E402
from app.routers import mojin_chat as mc  # noqa: E402

# 期望值（知识源由用户维护更新，数量偏差记 ⚠️ 不记 ❌）
EXPECTED_KB_MIN = 8        # kb_* 每语言最低块数
EXPECTED_PRODUCTS = 10     # products_* 每语言期望块数
LOCK_HINT_FILE_SIZE = 65536  # 打开成功但 list 为空且文件大于此值时视为可疑占用

OK, WARN, FAIL = "ok", "warn", "fail"


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════════
#  各项检查（独立失败互不影响，全部跑完再汇总退出码）
# ═══════════════════════════════════════════════════════════════════

def check_model_server() -> tuple[str, str]:
    """1. model server /healthz + 模型清单。"""
    import requests

    try:
        r = requests.get(f"{kb_settings.EMBEDDING_URL}/healthz", timeout=5)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        models = list(data.get("models", []))
        if status == "ok" and "bge-m3" in models and "bge-reranker-v2-m3" in models:
            return OK, f"status=ok, models=[{', '.join(models)}]"
        return FAIL, f"healthz 异常: status={status!r}, models={models}"
    except Exception as e:
        return FAIL, f"不可达: {type(e).__name__}"


def _open_milvus() -> tuple[MilvusManager | None, str | None]:
    """打开 Milvus Lite。返回 (manager, None) 或 (None, locked|other)。

    Milvus Lite 是文件锁：uvicorn 正在使用时，本进程打开会抛
    ConnectionConfigException（'Open local milvus failed'），此时返回 'locked'。
    """
    uri = kb_settings.MILVUS_URI
    try:
        mgr = MilvusManager(client=MilvusClient(uri=uri))
        return mgr, None
    except Exception as e:
        msg = str(e)
        if "milvus failed" in msg or "ConnectionConfigException" in msg:
            return None, "locked"
        return None, "other"


def check_milvus(mgr: MilvusManager | None, err: str | None) -> tuple[str, str]:
    """2. Milvus Lite 连通性（复用 main 中已打开的 mgr，避免锁窗口期内两次打开结果不一致）。"""
    if err is not None:
        if err == "locked":
            return WARN, "库文件无法打开（可能被运行中的 uvicorn 占用），跳过内容检查"
        return WARN, f"库文件打开异常（{err}）"
    try:
        cols = mgr.list_collections()
    except Exception as e:
        return WARN, f"list_collections 异常: {type(e).__name__}（可能库被占用）"
    if not cols:
        size = Path(kb_settings.MILVUS_URI).stat().st_size if Path(kb_settings.MILVUS_URI).exists() else 0
        if size > LOCK_HINT_FILE_SIZE:
            return WARN, f"库视图为空但文件 {size} 字节，疑似被其他进程占用或数据异常"
        return OK, "连通正常（空库，尚无集合）"
    return OK, f"连通正常，{len(cols)} 个集合"


def check_collections(mgr: MilvusManager | None, err: str | None) -> tuple[str, str]:
    """3. 6 集合完整性 + 实体数统计（复用同一 mgr）。"""
    expected = list(kb_settings.KB_COLLECTIONS.values()) + list(kb_settings.PRODUCT_COLLECTIONS.values())
    if err is not None:
        if err == "locked":
            return WARN, "库被 uvicorn 占用，无法统计集合（建议在服务停机时巡检内容）"
        return WARN, f"库文件打开异常（{err}），无法统计集合"
    try:
        cols = set(mgr.list_collections())
    except Exception as e:
        return WARN, f"list_collections 异常: {type(e).__name__}，无法统计集合"

    missing = [name for name in expected if name not in cols]
    if missing:
        return FAIL, f"集合缺失: {', '.join(missing)}（请运行 scripts/load_knowledge.py 摄入）"

    # 数量统计（query 取 id 计数）
    counts: dict[str, int] = {}
    try:
        for name in expected:
            rows = mgr.query(name, expr='id != ""', output_fields=["id"])
            counts[name] = len(rows)
    except Exception as e:
        return WARN, f"实体计数异常: {type(e).__name__}（可能库被占用）"

    deviations: list[str] = []
    for name in kb_settings.KB_COLLECTIONS.values():
        if counts[name] < EXPECTED_KB_MIN:
            deviations.append(f"{name}={counts[name]}(期望>={EXPECTED_KB_MIN})")
    for name in kb_settings.PRODUCT_COLLECTIONS.values():
        if counts[name] != EXPECTED_PRODUCTS:
            deviations.append(f"{name}={counts[name]}(期望=={EXPECTED_PRODUCTS})")

    detail = "、".join(f"{n}={c}" for n, c in counts.items())
    if deviations:
        return WARN, f"集合齐全，但数量偏差: {', '.join(deviations)}（知识源更新属正常）"
    return OK, f"6 集合齐全，数量符合预期（{detail}）"


def check_deepseek_key() -> tuple[str, str]:
    """4. DeepSeek key 存在性 + 极小调用验证（绝不打印 key 值）。"""
    key = (kb_settings.LLM_API_KEY or "").strip()
    if not key:
        return WARN, "DEEPSEEK_API_KEY 未设置（export 或写 MCPForge/.env 后重启生效），LLM 生成将走转人工兜底"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=kb_settings.LLM_URL, timeout=8.0)
        client.chat.completions.create(
            model=kb_settings.LLM_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return OK, "key 已设置，minimal 调用成功"
    except Exception as e:
        # 只报异常类型，不打印 key / 请求内容
        return WARN, f"key 已设置但调用失败: {type(e).__name__}"


def _smoke_search_products() -> tuple[str, str]:
    """search_products(query=gloves, lang=en) → results 非空。"""
    try:
        r = mc._run_search_products("gloves", "en")
    except Exception as e:
        return FAIL, f"调用异常: {type(e).__name__}"
    if r.get("results"):
        return OK, f"命中 {len(r['results'])} 条"
    if r.get("error"):
        return FAIL, f"链路失败: {r['error']}"
    return FAIL, "results 为空"


def _smoke_answer_question() -> tuple[str, str]:
    """answer_question(知识库外问题) → hit_count=0 且 answer 为转人工文案（兜底路径）。"""
    # 用确定库外的查询词测兜底；避免用 "cash on delivery" 等未来可能补进知识源的主题
    try:
        r = mc._run_answer("quantum entanglement", "en")
    except Exception as e:
        return FAIL, f"调用异常: {type(e).__name__}"
    if r.get("error"):
        return FAIL, f"链路失败: {r['error']}"
    if r.get("hit_count") != 0:
        return FAIL, f"期望无命中，实际 hit_count={r.get('hit_count')}（知识源可能新增了相关内容）"
    if r.get("answer") != mc.HUMAN_HANDOFF["en"]:
        return FAIL, "answer 不是转人工文案（兜底路径异常）"
    return OK, "hit_count=0，answer 为转人工文案"


def _smoke_order_help() -> tuple[str, str]:
    """order_help(query=shipping, lang=en) → answer/sources/hit_count/elapsed_ms 键齐全。"""
    try:
        r = mc._run_answer("shipping", "en", ["shipping", "support"])
    except Exception as e:
        return FAIL, f"调用异常: {type(e).__name__}"
    missing = sorted({"answer", "sources", "hit_count", "elapsed_ms"} - set(r))
    if missing:
        return FAIL, f"返回结构缺键: {missing}"
    note = f"，LLM 兜底({r['error']})" if r.get("error") else ""
    return OK, f"结构合法，hit_count={r.get('hit_count')}{note}"


def check_smoke(model_ok: bool, milvus_ok: bool) -> list[tuple[str, str, str]]:
    """5. 工具链路冒烟（真实链路；model server / Milvus 不可用时跳过记 ⚠️）。"""
    if not model_ok:
        reason = "model server 不可达，跳过冒烟"
    elif not milvus_ok:
        reason = "Milvus 库不可用（可能被 uvicorn 占用），跳过冒烟"
    else:
        return [
            ("smoke_search_products", *_smoke_search_products()),
            ("smoke_answer_question", *_smoke_answer_question()),
            ("smoke_order_help", *_smoke_order_help()),
        ]
    return [
        ("smoke_search_products", WARN, reason),
        ("smoke_answer_question", WARN, reason),
        ("smoke_order_help", WARN, reason),
    ]


# ═══════════════════════════════════════════════════════════════════
#  输出
# ═══════════════════════════════════════════════════════════════════

_STATUS_ICON = {OK: "✅", WARN: "⚠️", FAIL: "❌"}


def _human_output(checks: list[tuple[str, str, str]], elapsed_ms: int) -> str:
    lines = ["Mojin AI 客服全链路健康检查", f"时间: {_now_str()}", "─" * 60]
    for i, (name, status, detail) in enumerate(checks, 1):
        lines.append(f"{_STATUS_ICON[status]} [{i}/{len(checks)}] {name:<22} {detail}")
    lines.append("─" * 60)
    n_pass = sum(1 for _, s, _ in checks if s == OK)
    n_warn = sum(1 for _, s, _ in checks if s == WARN)
    n_fail = sum(1 for _, s, _ in checks if s == FAIL)
    elapsed = f"{elapsed_ms}ms" if elapsed_ms < 1000 else f"{elapsed_ms / 1000:.2f}s"
    lines.append(f"通过 {n_pass} | 告警 {n_warn} | 失败 {n_fail} | 总耗时 {elapsed}")
    return "\n".join(lines)


def _json_output(checks: list[tuple[str, str, str]], elapsed_ms: int) -> str:
    n_pass = sum(1 for _, s, _ in checks if s == OK)
    n_warn = sum(1 for _, s, _ in checks if s == WARN)
    n_fail = sum(1 for _, s, _ in checks if s == FAIL)
    payload = {
        "ok": n_fail == 0,
        "summary": {"pass": n_pass, "warn": n_warn, "fail": n_fail, "elapsed_ms": elapsed_ms},
        "checks": [{"name": name, "status": status, "detail": detail} for name, status, detail in checks],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mojin AI 客服全链路健康检查")
    parser.add_argument("--json", action="store_true", help="输出纯 JSON（给 cron/脚本用）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 关闭 loguru 默认 handler，避免应用日志混入输出（且不泄露 key/query）
    _loguru.remove()
    # 抑制第三方库（jieba / milvus-lite）的 logging 噪音，保证 --json 输出纯净
    import logging

    import jieba  # noqa: F401 确保已加载（jieba import 时会把 logger 设成 DEBUG）

    jieba.setLogLevel(logging.ERROR)
    logging.disable(logging.CRITICAL)  # milvus_lite 直接打 root logger（含 ERROR 级）

    t0 = time.perf_counter()
    checks: list[tuple[str, str, str]] = []

    # 统一打开一次 Milvus，第 2/3 项复用（milvus-lite 锁窗口期内两次打开可能结果不同）
    mgr, milvus_err = _open_milvus()

    checks.append(("model_server", *check_model_server()))
    model_ok = checks[-1][1] == OK

    checks.append(("milvus_lite", *check_milvus(mgr, milvus_err)))
    milvus_ok = checks[-1][1] == OK

    checks.append(("collections", *check_collections(mgr, milvus_err)))
    checks.append(("deepseek_key", *check_deepseek_key()))
    checks.extend(check_smoke(model_ok, milvus_ok))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if args.json:
        print(_json_output(checks, elapsed_ms))
    else:
        print(_human_output(checks, elapsed_ms))

    n_fail = sum(1 for _, s, _ in checks if s == FAIL)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
