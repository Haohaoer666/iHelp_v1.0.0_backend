#!/usr/bin/env bash
# 单入口拉起 FastAPI 应用（:8000）
set -euo pipefail
cd "$(dirname "$0")/.."

# 中文 Windows 默认 GBK，提前设置 UTF-8，避免读取中文文件出错。
export PYTHONUTF8=1

if [ ! -f .env ]; then
  echo "缺少 .env，请 cp .env.example .env 并填写配置" >&2
  exit 1
fi
set -a; source .env; set +a

uv run uvicorn app.main:app --port 8000
