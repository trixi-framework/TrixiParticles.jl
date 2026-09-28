# Surface reconstruction benchmarks

Deterministic micro- and meso-scale benchmarks for `SurfaceReconstruction`. No extra
dependencies beyond the package itself: timings use `time_ns`, allocations use
`@allocated`.

## Workloads

- `block_small`: 7³ lattice, the numerical reference geometry (sanity + smoke).
- `block_1e4`: ~22³ lattice; covers the full pipeline at a still-interactive scale.
- `boundaries`: 7³ lattice plus one lattice boundary cube exercising the BVH
  signed-distance constraint path.
- `sparse`: 7³ lattice plus an isolated droplet with the sparse-component fallback
  enabled, exercising the graph search and fallback-mesh path.
- `wave_tank` (opt-in): synthetic production-scale frame with 1.1e6 particles in a
  2 m × 4 m × 1 m tank at the particle spacing of the production wave tank
  (599×1180×309 grid). Every repeat restarts the isovalue correction at the base
  isovalue, so it times the full correction of a new frame. The workspace needs about
  3.5 GB of memory (5 GB peak); run it with all available threads.

The default suite runs all workloads except `wave_tank` in well under a minute
(excluding one-time compilation).

Workloads reuse one `SurfaceReconstruction` across warmup and repeats, so timings reflect
the warm-started steady state (typically a single correction evaluation on identical
input). Short workloads are evaluated several times per timing sample. This keeps the
regression signal low-noise; cold-start behavior is covered by the numerical reference tests.

The `sparse` workload has no warm-started steady state: its second correction pass
excludes the droplet from the target volume, so every call runs both passes and ends at a
slightly different isovalue within the volume tolerance. The number of calls per run is
fixed, so the reported results are still reproducible.

## Run locally

```bash
julia --project=benchmark/surface_reconstruction \
    benchmark/surface_reconstruction/run_benchmarks.jl \
    --workloads=block_small,block_1e4 --repeats=3 --output=results.json
```

Optionally record a local baseline, then compare against it (exit code 1 on regression
beyond `--tolerance` relative time or 5% allocation growth). Workstation-specific
baselines are ignored by git; the runner warns when thread counts differ.

```bash
julia --project=benchmark/surface_reconstruction \
    benchmark/surface_reconstruction/run_benchmarks.jl \
    --output=benchmark/surface_reconstruction/baselines/dev-workstation-julia1.11.json
julia --project=benchmark/surface_reconstruction \
    benchmark/surface_reconstruction/run_benchmarks.jl \
    --baseline=benchmark/surface_reconstruction/baselines/dev-workstation-julia1.11.json
```

Production-scale timing:

```bash
julia --project=benchmark/surface_reconstruction --threads=auto \
    benchmark/surface_reconstruction/run_benchmarks.jl --workloads=wave_tank
```

For cross-thread scaling, set `JULIA_NUM_THREADS` (bitwise-identical meshes are required
at 1 and N threads; see the determinism guard in the test suite).

## Determinism check

`benchmark/surface_reconstruction/determinism.jl` prints SHA-256 mesh checksums and
selected statistics for a reference block and two wave tanks. The larger case also
exercises fixed-bucket mesh reductions. The output must agree byte-for-byte across
thread counts on the same platform
(required by pipelines that hash outputs for auditing). A local baseline is a developer
regression reference for its machine, not a portable CI threshold.

```bash
JULIA_NUM_THREADS=1 julia --project=benchmark/surface_reconstruction \
    benchmark/surface_reconstruction/determinism.jl > det1.txt
JULIA_NUM_THREADS=4 julia --project=benchmark/surface_reconstruction \
    benchmark/surface_reconstruction/determinism.jl > det4.txt
diff <(grep -v "^threads=" det1.txt) <(grep -v "^threads=" det4.txt) && echo DETERMINISTIC
```
