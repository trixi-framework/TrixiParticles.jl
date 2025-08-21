#!/usr/bin/env bash
set -euo pipefail

# ─── PARSE OPTIONS ────────────────────────────────────────────────────────────
TEST_MODE=0
if [[ "${1:-}" == "--test" ]]; then
  TEST_MODE=1
  shift
fi

# ─── USER CONFIG (via environment variables) ──────────────────────────────────
# e.g. export TRIXI_SUBMIT_PROJECT_PATH="/my/project"; then run this script

TRIXI_SUBMIT_PROJECT_PATH="${TRIXI_SUBMIT_PROJECT_PATH:-/path/to/your/project}"
TRIXI_SUBMIT_PY_GENERATOR="${TRIXI_SUBMIT_PY_GENERATOR:-generate_commands.py}"
TRIXI_SUBMIT_JULIA_SRC="${TRIXI_SUBMIT_JULIA_SRC:-run_validation.jl}"
TRIXI_SUBMIT_COMMANDS_FILE="${TRIXI_SUBMIT_COMMANDS_FILE:-commands.txt}"
TRIXI_SUBMIT_WORKDIR="${TRIXI_SUBMIT_WORKDIR:-/path/to/working/directory}"
TRIXI_SUBMIT_MPI_TASKS="${TRIXI_SUBMIT_MPI_TASKS:-1}"
TRIXI_SUBMIT_THREADS="${TRIXI_SUBMIT_THREADS:-48}"
TRIXI_SUBMIT_OUTPUT_DIR="${TRIXI_SUBMIT_OUTPUT_DIR:-~/slurm_output}"
TRIXI_SUBMIT_SBATCH_TIME="${TRIXI_SUBMIT_SBATCH_TIME:-23:00:00}"
TRIXI_SUBMIT_SBATCH_PARTITION="${TRIXI_SUBMIT_SBATCH_PARTITION:-pNode}"
# ─── END USER CONFIG ───────────────────────────────────────────────────────────
# regenerate commands; python emits "1" or "X"
env TRIXI_SUBMIT_THREADS="$TRIXI_SUBMIT_THREADS" \
    python3 "$TRIXI_SUBMIT_PY_GENERATOR" \
    "$TRIXI_SUBMIT_JULIA_SRC" \
    "$TRIXI_SUBMIT_COMMANDS_FILE"

mkdir -p "$TRIXI_SUBMIT_OUTPUT_DIR"

declare -a _threads_arr
declare -a _cmds_arr

# use prefix to extract meaningful part for job names
trixi_prefix='trixi_include(@__MODULE__, joinpath(validation_dir(),'

while IFS=$'\t' read -r tag cmd; do
  [[ -z "$cmd" || "${cmd:0:1}" == "#" ]] && continue

  # determine threads
  if [[ "$tag" == "X" ]]; then
    threads="$TRIXI_SUBMIT_THREADS"
  else
    threads="$tag"
  fi

    # extract substring after prefix for naming
  if [[ "$cmd" == *"$trixi_prefix"* ]]; then
    body="${cmd#*${trixi_prefix}}"
  else
    body="$cmd"
  fi

  # sanitize job name
  name=$(echo "$body" \
    | sed -e 's/[^[:alnum:]]\+/_/g' -e 's/^_//' -e 's/_$//')

  # properly escape double quotes in the Julia command for --eval
  escaped_cmd="${cmd//\"/\\\"}"

  if [[ "$TEST_MODE" -eq 1 ]]; then
    _threads_arr+=("$threads")
    _cmds_arr+=("$cmd")
  else
    sbatch \
      --job-name="$name" \
      --output="${TRIXI_SUBMIT_OUTPUT_DIR}/${name}.out" \
      --error="${TRIXI_SUBMIT_OUTPUT_DIR}/${name}.err" \
      --ntasks="$TRIXI_SUBMIT_MPI_TASKS" \
      --cpus-per-task="$threads" \
      --time="$TRIXI_SUBMIT_SBATCH_TIME" \
      --partition="$TRIXI_SUBMIT_SBATCH_PARTITION" \
      <<EOF
#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$TRIXI_SUBMIT_WORKDIR/$name"
cd "$TRIXI_SUBMIT_WORKDIR/$name"
echo "Running command: $escaped_cmd"

julia --project="$TRIXI_SUBMIT_PROJECT_PATH" \
           -t "$threads" \
           --eval "using TrixiParticles; $escaped_cmd"
EOF
  fi

done < "$TRIXI_SUBMIT_COMMANDS_FILE"

if [[ "$TEST_MODE" -eq 1 ]]; then
  cat <<EOF
Configuration settings:
  TRIXI_SUBMIT_PROJECT_PATH  = $TRIXI_SUBMIT_PROJECT_PATH
  TRIXI_SUBMIT_JULIA_SRC      = $TRIXI_SUBMIT_JULIA_SRC
  TRIXI_SUBMIT_COMMANDS_FILE  = $TRIXI_SUBMIT_COMMANDS_FILE
  TRIXI_SUBMIT_WORKDIR        = $TRIXI_SUBMIT_WORKDIR
  TRIXI_SUBMIT_MPI_TASKS      = $TRIXI_SUBMIT_MPI_TASKS
  TRIXI_SUBMIT_THREADS        = $TRIXI_SUBMIT_THREADS
  TRIXI_SUBMIT_OUTPUT_DIR     = $TRIXI_SUBMIT_OUTPUT_DIR
  TRIXI_SUBMIT_SBATCH_TIME    = $TRIXI_SUBMIT_SBATCH_TIME
  TRIXI_SUBMIT_SBATCH_PARTITION = $TRIXI_SUBMIT_SBATCH_PARTITION
EOF
  echo
  echo "Planned jobs (CPUS_PER_TASK | Command):"
  {
    printf "%s\t%s\n" "CPUS" "COMMAND"
    for i in "${!_threads_arr[@]}"; do
      printf "%s\t%s\n" "${_threads_arr[i]}" "${_cmds_arr[i]}"
    done
  } | column -t -s$'\t'
fi