#!/usr/bin/env bash
# Push the airportaccess repo to the Featurize GPU server and (optionally) install deps + run a command.
#
# Usage:
#   FEATURIZE_PASS=<password> scripts/deploy_featurize.sh push
#   FEATURIZE_PASS=<password> scripts/deploy_featurize.sh install
#   FEATURIZE_PASS=<password> scripts/deploy_featurize.sh run "python scripts/run_sanity.py"
#   FEATURIZE_PASS=<password> scripts/deploy_featurize.sh full "python scripts/train_risk_field.py --model mlp --airport KLAX"
#
# Aliyun mirror is used for pip to make installs fast inside CN.
set -euo pipefail

FEATURIZE_USER="${FEATURIZE_USER:-featurize}"
FEATURIZE_HOST="${FEATURIZE_HOST:-workspace.featurize.cn}"
FEATURIZE_PORT="${FEATURIZE_PORT:-57925}"
REMOTE_DIR="${REMOTE_DIR:-/home/featurize/work/airportaccess}"

if [[ -z "${FEATURIZE_PASS:-}" ]]; then
  echo "FEATURIZE_PASS must be set (the SSH password)." >&2
  exit 2
fi

SSH_CMD=(sshpass -e ssh -o StrictHostKeyChecking=no -o BatchMode=no -p "$FEATURIZE_PORT" "$FEATURIZE_USER@$FEATURIZE_HOST")
RSYNC_CMD=(sshpass -e rsync -az --delete --stats
  --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/'
  --exclude '.git/' --exclude 'data/cache/' --exclude 'data/raw/' --exclude 'data/processed/'
  --exclude 'results/' --exclude 'models/' --exclude 'figures/' --exclude '.DS_Store'
  -e "ssh -o StrictHostKeyChecking=no -p $FEATURIZE_PORT")
PIP="/environment/miniconda3/bin/pip install --timeout 120 --retries 5 -i https://mirrors.aliyun.com/pypi/simple/"

cmd="${1:-help}"; shift || true

case "$cmd" in
  probe)
    SSHPASS="$FEATURIZE_PASS" "${SSH_CMD[@]}" \
      "hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; /environment/miniconda3/bin/python --version"
    ;;
  push)
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    SSHPASS="$FEATURIZE_PASS" "${RSYNC_CMD[@]}" "$REPO_ROOT/" "$FEATURIZE_USER@$FEATURIZE_HOST:$REMOTE_DIR/"
    ;;
  install)
    SSHPASS="$FEATURIZE_PASS" "${SSH_CMD[@]}" \
      "cd $REMOTE_DIR && $PIP -r requirements.txt"
    ;;
  run)
    [[ $# -ge 1 ]] || { echo "run requires a command argument"; exit 2; }
    RUNCMD="$*"
    SSHPASS="$FEATURIZE_PASS" "${SSH_CMD[@]}" \
      "cd $REMOTE_DIR && PYTHONPATH=. /environment/miniconda3/bin/python -u -c 'import sys; print(\"py=\", sys.version)' && $RUNCMD"
    ;;
  full)
    [[ $# -ge 1 ]] || { echo "full requires a command argument"; exit 2; }
    RUNCMD="$*"
    bash "$0" push
    bash "$0" install
    bash "$0" run "$RUNCMD"
    ;;
  pull)
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    SSHPASS="$FEATURIZE_PASS" sshpass -e rsync -az --stats \
      -e "ssh -o StrictHostKeyChecking=no -p $FEATURIZE_PORT" \
      "$FEATURIZE_USER@$FEATURIZE_HOST:$REMOTE_DIR/results/" "$REPO_ROOT/results/"
    SSHPASS="$FEATURIZE_PASS" sshpass -e rsync -az --stats \
      -e "ssh -o StrictHostKeyChecking=no -p $FEATURIZE_PORT" \
      "$FEATURIZE_USER@$FEATURIZE_HOST:$REMOTE_DIR/models/" "$REPO_ROOT/models/" || true
    ;;
  help|*)
    sed -n '1,16p' "$0"
    ;;
esac
