#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TASK_SPEC=""
SEED_START=""
SEED_END=""
MAX_JOBS=6
DEVICE="auto"
MODE="top"
LOG_ROOT="logs"
SAVE_ROOT="saved_models"
TOTAL_TIMESTEPS=""

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_sac_time_aware_batch.sh [options]

Required:
  --tasks TASKS          One task, comma-separated tasks, or "all".
  --seed-start N         First seed, inclusive.
  --seed-end N           Last seed, inclusive.

Options:
  --max-jobs N           Concurrent seeds within each task (default: 6).
  --device DEVICE        auto, cpu, cuda, or mps (default: auto).
  --mode MODE            Hyperparameter mode (default: top).
  --log-root PATH        Training log root (default: logs).
  --save-root PATH       Saved-model root (default: saved_models).
  --total-timesteps N    Optional smoke-test override.
  -h, --help             Show this help.

Examples:
  # WSL: one task, six local seeds
  bash benchmarks/run_sac_time_aware_batch.sh \
    --tasks cheetah-run --seed-start 0 --seed-end 5 \
    --max-jobs 6 --device cuda

  # Mac: all tasks, six Mac-owned seeds
  bash benchmarks/run_sac_time_aware_batch.sh \
    --tasks all --seed-start 6 --seed-end 11 \
    --max-jobs 6 --device mps
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      TASK_SPEC="${2:-}"
      shift 2
      ;;
    --seed-start)
      SEED_START="${2:-}"
      shift 2
      ;;
    --seed-end)
      SEED_END="${2:-}"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="${2:-}"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="${2:-}"
      shift 2
      ;;
    --total-timesteps)
      TOTAL_TIMESTEPS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK_SPEC" || -z "$SEED_START" || -z "$SEED_END" ]]; then
  echo "--tasks, --seed-start, and --seed-end are required." >&2
  usage >&2
  exit 2
fi

if ! [[ "$SEED_START" =~ ^[0-9]+$ && "$SEED_END" =~ ^[0-9]+$ ]]; then
  echo "Seed bounds must be non-negative integers." >&2
  exit 2
fi
if (( SEED_END < SEED_START )); then
  echo "--seed-end must be greater than or equal to --seed-start." >&2
  exit 2
fi
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-jobs must be a positive integer." >&2
  exit 2
fi
if [[ -n "$TOTAL_TIMESTEPS" ]] && ! [[ "$TOTAL_TIMESTEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--total-timesteps must be a positive integer." >&2
  exit 2
fi

ALL_TASKS=(humanoid-walk cheetah-run walker-run quadruped-run trading)
if [[ "$TASK_SPEC" == "all" ]]; then
  TASKS=("${ALL_TASKS[@]}")
else
  IFS=',' read -r -a TASKS <<< "$TASK_SPEC"
fi

for TASK in "${TASKS[@]}"; do
  VALID=false
  for KNOWN_TASK in "${ALL_TASKS[@]}"; do
    if [[ "$TASK" == "$KNOWN_TASK" ]]; then
      VALID=true
      break
    fi
  done
  if [[ "$VALID" != true ]]; then
    echo "Unknown task: $TASK" >&2
    echo "Valid tasks: ${ALL_TASKS[*]}" >&2
    exit 2
  fi
done

cd "$REPO_ROOT"

RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LAUNCH_ROOT="$LOG_ROOT/launchers/sac_time_aware/$RUN_STAMP"
mkdir -p "$LAUNCH_ROOT"

echo "SAC time-aware batch"
echo "repo=$REPO_ROOT"
echo "tasks=${TASKS[*]}"
echo "seeds=$SEED_START..$SEED_END"
echo "max_jobs=$MAX_JOBS"
echo "device=$DEVICE"
echo "launcher_logs=$LAUNCH_ROOT"
echo "started=$(date)"

ACTIVE_PIDS=()
ACTIVE_LOGS=()
ACTIVE_SEEDS=()

stop_active_jobs() {
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    echo "Stopping active seed processes..." >&2
    for PID in "${ACTIVE_PIDS[@]}"; do
      kill "$PID" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
}
trap 'stop_active_jobs; exit 130' INT TERM

wait_for_batch() {
  local BATCH_STATUS=0
  local INDEX PID LOG_PATH SEED

  for INDEX in "${!ACTIVE_PIDS[@]}"; do
    PID="${ACTIVE_PIDS[$INDEX]}"
    LOG_PATH="${ACTIVE_LOGS[$INDEX]}"
    SEED="${ACTIVE_SEEDS[$INDEX]}"

    if ! wait "$PID"; then
      echo "seed=$SEED exited with an error; see $LOG_PATH" >&2
      BATCH_STATUS=1
    elif ! grep -q "Training finished." "$LOG_PATH"; then
      echo "seed=$SEED has no completion marker; see $LOG_PATH" >&2
      BATCH_STATUS=1
    else
      echo "seed=$SEED completed"
    fi
  done

  ACTIVE_PIDS=()
  ACTIVE_LOGS=()
  ACTIVE_SEEDS=()
  return "$BATCH_STATUS"
}

for TASK in "${TASKS[@]}"; do
  TASK_LOG_DIR="$LAUNCH_ROOT/$TASK"
  mkdir -p "$TASK_LOG_DIR"
  echo "task=$TASK started=$(date)"
  TASK_STATUS=0

  for SEED in $(seq "$SEED_START" "$SEED_END"); do
    LOG_PATH="$TASK_LOG_DIR/seed_${SEED}.log"
    COMMAND=(
      python -m benchmarks.run_discrete_rl
      --algos sac_time_aware
      --env_id "$TASK"
      --mode "$MODE"
      --seed "$SEED"
      --device "$DEVICE"
      --log_root "$LOG_ROOT"
      --save_root "$SAVE_ROOT"
    )
    if [[ -n "$TOTAL_TIMESTEPS" ]]; then
      COMMAND+=(--total_timesteps "$TOTAL_TIMESTEPS")
    fi

    "${COMMAND[@]}" > "$LOG_PATH" 2>&1 &
    ACTIVE_PIDS+=("$!")
    ACTIVE_LOGS+=("$LOG_PATH")
    ACTIVE_SEEDS+=("$SEED")
    echo "seed=$SEED pid=$! log=$LOG_PATH"

    if (( ${#ACTIVE_PIDS[@]} >= MAX_JOBS )); then
      wait_for_batch || TASK_STATUS=1
    fi
  done

  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    wait_for_batch || TASK_STATUS=1
  fi

  echo "task=$TASK status=$TASK_STATUS finished=$(date)"
  if (( TASK_STATUS != 0 )); then
    echo "Stopping before the next task." >&2
    exit 1
  fi
done

echo "all requested tasks completed at $(date)"
