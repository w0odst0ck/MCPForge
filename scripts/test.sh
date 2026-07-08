#!/usr/bin/env bash
# ── 测试 ──
uv run pytest tests/ -v --cov=app --cov-report=term-missing
