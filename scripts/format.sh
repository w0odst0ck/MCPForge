#!/usr/bin/env bash
# ── 代码格式化 ──
ruff format app/ tests/
ruff check --fix app/ tests/
