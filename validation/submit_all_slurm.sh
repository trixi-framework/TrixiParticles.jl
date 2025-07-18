```bash
#!/usr/bin/env bash
set -euo pipefail

# ─── USER CONFIG (via environment variables) ──────────────────────────────────
# e.g. export TRIXI_SUBMIT_PROJECT_PATH="/my/project"; then run this script

TRIXI_SUBMIT_PROJECT_PATH="${TRIXI_SUBMIT_PROJECT_PATH:-/path/to/your/project}"
TRIXI_SUBMIT_PY_GENERATOR="${TRIXI_SUBMIT_PY_GENERATOR:-generate_commands.py}"
TRIXI_SUBMIT_JULIA_SRC="${TRIXI_SUBMIT_JULIA_SRC:-validation.jl}"
TRIXI_SUBMIT_COMMANDS_FILE="${TRIXI_SUBMIT_COMMANDS_FILE:-commands.txt}"
TRIXI_SUBMIT_WORKDIR="${TRIXI_SUBMIT_WORKDIR:-/path/to/working/directory}"
TRIXI_SUBMIT_MPI_TASKS="${TRIXI_SUBMIT_MPI_TASKS:-1}"
TRIXI_SUBMIT_THREADS="${TRIXI_SUBMIT_THREADS:-48}"
TRIXI_SUBMIT_OUTPUT_DIR="${TRIXI_SUBMIT_OUTPUT_DIR:-~/slurm_output}"
TRIXI_SUBMIT_SBATCH_TIME="${TRIXI_SUBMIT_SBATCH_TIME:-23:00:00}"
TRIXI_SUBMIT_SBATCH_PARTITION="${TRIXI_SUBMIT_SBATCH_PARTITION:-pNode}"
# ─── END USER CONFIG ───────────────────────────────────────────────────────────

# regenerate the list of trixi_include(...) commands with thread counts
python3 "$TRIXI_SUBMIT_PY_GENERATOR" \
    "$TRIXI_SUBMIT_JULIA_SRC" \
    "$TRIXI_SUBMIT_COMMANDS_FILE"

mkdir -p "$TRIXI_SUBMIT_OUTPUT_DIR"

while IFS=$'\t' read -r threads cmd; do
  # skip blank lines or comments
  [[ -z "$cmd" || "${cmd:0:1}" == "#" ]] && continue

  # make a safe job name from the command
  name=$(echo "$cmd" \
    | sed -e 's/[^[:alnum:]]\+/_/g' \
          -e 's/^_//' -e 's/_$//')

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

# create and switch into a per‐job working directory
mkdir -p "$TRIXI_SUBMIT_WORKDIR/$name"
cd "$TRIXI_SUBMIT_WORKDIR/$name"

# run the MPI‑Julia job with the assigned thread count
srun julia --project="$TRIXI_SUBMIT_PROJECT_PATH" \
           -t "$threads" \
           --eval "using TrixiParticles; $cmd"
EOF

done < "$TRIXI_SUBMIT_COMMANDS_FILE"
```
