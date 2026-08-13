"""Mojin 知识摄入 CLI。

用法（在 MCPForge 目录下）:
    uv run python scripts/load_knowledge.py [--rebuild] [--lang en,cn,ar] [--scope kb|products|all]
    uv run python scripts/load_knowledge.py --check-only [--lang en,cn,ar] [--scope kb|products|all]

前置:
    - model server 运行于 127.0.0.1:9997（/embeddings）
    - 知识源目录 carewell-shop 存在（MOJIN_CAREWELL_PATH 可覆盖）

--rebuild:    先 drop 再重建集合（默认幂等 upsert）
--scope:      kb（FAQ+政策）| products（产品）| all（默认）
--check-only: 只预览不写库：读取知识源分块，打印各语言块数与库中现有块数的差异
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从任意 cwd 都能 import app 包（uv run 在 MCPForge 目录下执行）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.knowledge.loader import load_knowledge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mojin 知识摄入（FAQ/政策/产品 → Milvus Lite）")
    parser.add_argument("--rebuild", action="store_true", help="先 drop 再重建集合")
    parser.add_argument("--lang", default="en,cn,ar", help="语言列表，逗号分隔，如 en,cn,ar")
    parser.add_argument("--scope", default="all", choices=["kb", "products", "all"], help="摄入范围")
    parser.add_argument("--check-only", action="store_true", help="只预览不写库：打印分块数与库中差异")
    return parser.parse_args()


def _validate_langs(langs: list[str]) -> bool:
    from app.knowledge.loader import LANGS

    bad = [lang for lang in langs if lang not in LANGS]
    if bad:
        print(f"❌ 非法语言: {bad}（必须是 {'/'.join(LANGS)}）")
        return False
    return True


def run_check_only(langs: list[str], scope: str) -> int:
    """--check-only：不写库，只读取知识源分块并打印各语言块数与库中差异（知识源更新前预览）。"""
    from app.knowledge.config import kb_settings
    from app.knowledge.loader import _collect_blocks, default_manager

    carewell = Path(kb_settings.CAREWELL_PATH)
    if not carewell.is_dir():
        print(f"❌ 知识源目录不存在（可设 MOJIN_CAREWELL_PATH 覆盖）: {carewell}")
        return 1

    # 库中现有块数（读库；被 uvicorn 占用时显示 n/a，不崩溃）
    db_counts: dict[str, int] | None = {}
    try:
        mgr = default_manager()
        for col in (*kb_settings.KB_COLLECTIONS.values(), *kb_settings.PRODUCT_COLLECTIONS.values()):
            try:
                db_counts[col] = len(mgr.query(col, expr='id != ""', output_fields=["id"]))
            except Exception:
                db_counts[col] = 0  # 集合不存在 → 视为 0 块
    except Exception as e:
        print(f"⚠️ 库无法连接（可能被 uvicorn 占用）：{type(e).__name__}，库中块数显示 n/a")
        db_counts = None

    print(f"预览（不写库）| scope={scope} langs={','.join(langs)}")
    total_src = 0
    total_diff = 0
    for lang in langs:
        blocks_by_kind = _collect_blocks(carewell, lang, scope)
        for kind, blocks in blocks_by_kind.items():
            col = kb_settings.kb_collection(lang) if kind == "kb" else kb_settings.product_collection(lang)
            src = len(blocks)
            total_src += src
            db = db_counts.get(col) if db_counts is not None else None
            if db is None:
                print(f"  {col}: 源 {src} 块 | 库中 n/a")
                continue
            diff = src - db
            total_diff += diff
            mark = "🟢 无变化" if diff == 0 else "🟡 需更新"
            print(f"  {col}: 源 {src} 块 | 库中 {db} 块 | 差异 {diff:+d} {mark}")
    print(f"\n共 {len(langs)} 语言 × scope={scope}，源合计 {total_src} 块，净差异 {total_diff:+d}")
    return 0


def main() -> int:
    args = parse_args()
    langs = [item.strip() for item in args.lang.split(",") if item.strip()]
    if not _validate_langs(langs):
        return 1
    if args.check_only:
        return run_check_only(langs, args.scope)
    print(f"开始摄入 | scope={args.scope} langs={langs} rebuild={args.rebuild}")
    counts = load_knowledge(langs=langs, scope=args.scope, rebuild=args.rebuild)
    print("\n各集合写入块数：")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    total = sum(counts.values())
    print(f"\n✅ 摄入完成，共写入 {total} 块（{len(counts)} 个集合）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
