# Reusable whitewater and Blender rendering tools

`pipeline.py` exposes two independent stages for **3D particle-fluid cases**:

1. `foam`: convert saved fluid VTK positions/velocities to Partio, run the pinned
   SPlisHSPlasH FoamGenerator sequentially, and package foam, spray, and bubbles
   as velocity-bearing PLY frames.
2. `render`: render PR #1329 surfaces, tracked solids, and optionally these
   secondary particles with Blender. It also accepts a pre-colored stress PLY.

There is **no case configuration file**. Every case, model, material, camera,
lighting, and output choice is a CLI parameter. `*.pvd` and `*.json` inputs
referenced below are *generated data inventories/provenance*, not configuration
files. `python tools/rendering/pipeline.py foam --help` and `render --help`
enumerate every option. Use a new output directory for each run.

This PR adds reusable tools stacked on the #1333 simulation/reconstruction
integration. Migrating the **existing v03 media-production pipeline and its
approved assets** is a follow-up PR. No v03 release files or press scripts are
rewritten here.

## Prerequisites

- Fluid VTK time series (`fluid_1.pvd` or any other name) with 3D point
  coordinates and a velocity point array. Saved times must be uniformly spaced.
- A surface reconstruction: the native PR #1329 PLY + VTP/PVD output, or a
  generated metadata file listing `frames`, each with `source_timestep`,
  `simulation_time_s`, `water_file`, and `blade_files`.
- Blender 5.2 with the Volume Coefficients and Geometry Nodes used here.
- Foam stage only: Python 3.9, `meshio==5.3.5`, `numpy==2.0.2`,
  `partio==1.0.0`, and a FoamGenerator binary built with the patch below.
  The render stage is launched by ordinary Python and uses Blender's own Python
  and bundled NumPy; it does not import Partio or meshio.

The patch in `build/` targets SPlisHSPlasH `f3f677140761db7637b5443beb54f19f1f835ed4`
(2.18.1). It supports a headless build, explicit `--seed`, velocity arrays in
split secondary-particle VTK, and zero-range potential normalization. When a
calibrated generation potential has no range, its contribution is zero rather
than `NaN`; this changes only the zero-range case. Build `FoamGenerator` with
the bundled script, which pins the upstream commit, applies the patch once,
and configures double precision (all inputs are CLI parameters):

```bash
python tools/rendering/build/build_foam_generator.py \
  --source-dir /path/to/SPlisHSPlasH \
  --build-dir /path/to/foam-build \
  --jobs 4
```

It needs `git`, `cmake`, and `ninja` on `PATH` (for example via
`uv run --with cmake --with ninja`). Re-running the script against the same directories is idempotent. The
executable appears at `/path/to/SPlisHSPlasH/bin/FoamGenerator`. Its `--help`
output must include `--seed`; the wrapper rejects older binaries.

## 1. Generate whitewater

An illustrative invocation for a fluid inside a translated 3D domain:

```bash
uv run --no-project --python 3.9 \
  --with meshio==5.3.5 --with numpy==2.0.2 --with partio==1.0.0 \
  python tools/rendering/pipeline.py foam \
  --fluid-pvd /path/to/fluid.pvd \
  --generator /path/to/SPlisHSPlasH/bin/FoamGenerator \
  --generator-source /path/to/SPlisHSPlasH/Tools/FoamGenerator/main.cpp \
  --generator-patch tools/rendering/build/splishsplash_foam_generator.patch \
  --output /path/to/new/whitewater \
  --domain-min -1 0 0 --domain-max 2 2 1 \
  --generator-radius 0.01 --foam-scale 1000 \
  --lifetime-min 2 --lifetime-max 5 --buoyancy 2 \
  --drag 0.8 --drag-reference-step 0.02 \
  --seed 144 --generator-threads 1 --velocity-array velocity \
  --output-axis-order xyz
```

The generator always reads the **complete specified PVD sequence** in physical
order before producing the requested output; its automatic potential limits
are calibrated on that complete sequence. Choose a render frame in stage 2,
not by starting the generator in the middle of an evolving event.

`foam_metadata.json` records the source timestamps/hashes, generator binary,
seed, calibrated log, category counts, clipping against the supplied domain,
the effective drag after cadence conversion, and output hashes. A complete
cache may be reused with the same command plus `--reuse`; it validates inputs
and every output. Incomplete generation must be restarted into a fresh
directory, since resetting particle history at a later frame changes the
model. The output PLY contains position and velocity in the specified axis
order, even when only one of the three secondary types is present.

## 2. Render surfaces and optional whitewater

Pass one or more `--solid-pattern` arguments if the case has moving structures.
Use `{iter}` for the *actual zero-padded suffix from the native surface PVD*
(`surface_structure_1_{iter}.ply`), or `{frame:06d}` for a saved-state index.
For a generated reconstruction metadata file, the solids listed by its frame
records are used directly. The water and foam input axis orders are separate:

- PR #1329's native PLY is in simulation `xyz`, so use `--surface-axis-order xyz`.
- Existing Blender-coordinate PLYs use `xzy`; selecting it prevents a second
  axis swap. The generated foam stage records its chosen order and the renderer
  checks it. Vertices, velocities, mesh winding, and normals must be converted
  consistently; the Blender worker flips imported face winding after an `xyz`
  reflection.

Example render arguments for an illustrative one-frame liquid/whitewater case
(replace paths and numbers for the current scene):

```bash
python tools/rendering/pipeline.py render \
  --blender /path/to/blender --surface-dir /path/to/surface \
  --surface-pvd /path/to/surface/surface_fluid_1.pvd \
  --solid-pattern 'surface_structure_1_{iter}.ply' \
  --foam-dir /path/to/new/whitewater \
  --output /path/to/new/frame_000000.png --frame 0 \
  --surface-axis-order xyz --foam-axis-order xyz \
  --camera-position 3 2 2 --camera-target 1 0.5 0.3 --camera-fov 34 \
  --tank-min 0 0 0 --tank-max 2 4 1 \
  --light 'key:0.2:2.8:3.8:1250:1:0.76:0.58:3.2' \
  --width 640 --height 360 --samples 16 --device cpu \
  --world-color 0.002 0.005 0.012 --world-strength 0.16 \
  --water-color 0.35 0.72 1 --water-roughness 0.06 --water-ior 1.333 \
  --water-absorption 0.85 0.14125 0.02305 \
  --water-scattering 0.02 0.04 0.08 --water-scattering-anisotropy 0.35 \
  --blade-color 0.62 0.10 0.018 --blade-metallic 0.92 \
  --blade-roughness 0.30 --blade-coat 0.16 \
  --blade-floor-extension 0 --blade-bevel 0.005 \
  --floor-color 0.02 0.03 0.05 --floor-metallic 0.3 --floor-roughness 0.4 \
  --glass-color 0.06 0.42 0.34 --glass-opacity 0.04 \
  --glass-roughness 0.30 --glass-ior 1.36 \
  --wall-thickness 0.018 --floor-thickness 0.018 --visible-wall-height 0.16 \
  --foam-color 0.72 0.82 0.86 --foam-roughness 0.46 \
  --foam-transmission 0.04 --foam-subsurface 0.035 \
  --foam-point-radius 0.006 --foam-voxel-size 0.004 \
  --foam-threshold 0.5 --foam-adaptivity 0.12 \
  --foam-noise-scale 220 --foam-noise-detail 3 \
  --foam-noise-roughness 0.7 --foam-micro-bump 0.00045 \
  --foam-bump-strength 0.22 \
  --spray-color 0.82 0.94 1 --spray-roughness 0.025 \
  --spray-radius 0.0008 --spray-scale-range 0.55 1.15 \
  --spray-exposure 0.001 --spray-max-stretch 4 \
  --bubble-color 0.9 0.97 1 --bubble-roughness 0.025 \
  --bubble-radius 0.0014 --bubble-scale-range 0.35 1.75 \
  --bubble-relative-ior 0.75
```

`render --sequence --start 0 --stop 10 --stride 1` writes numbered PNGs plus
`render_metadata.json` and `render_progress.json`. `--resume` validates the
settings, source hashes, renderer source, and PNG prefix before continuing.
An already complete sequence cannot be resumed or silently overwritten.

For a pre-colored stress PLY, use `--mode stress --stress-mesh FILE`, optionally
`--stress-water-opacity` and `--stress-normal-light`. The v03 stress image uses
an independently color-encoded YlOrRd PLY; this tool preserves the input
vertex colors, its stress emission material, and the contextual ghost-water
and floor treatments. Stress export itself remains an upstream input. When
`--stress-water-opacity` is zero, the stress mode does **not** require unused
liquid, glass, blade, or whitewater material parameters. When contextual water
is shown, supply `--water-color`, `--water-roughness`, and `--water-ior`; its
ghosted shader is separate from the primary liquid-volume shader.

## Materials and scientific scope

There is **one** physically based transmitted/scattered liquid shader, with
all coefficients supplied by CLI flags. Optional foam, velocity-aligned spray,
and bubble interfaces use their own parameters. Solids, glass, the floor, and
the stress view retain configurable materials. The rejected `legacy-cyan`
and `physical-clear` water branches and the redundant `foaming-water` preset
are not implemented. Enabling whitewater changes the scene, not the liquid
material. Camera, lights, render engine/device, color management, and all
render-only geometry offsets are explicit flags.

Whitewater is a heuristic visualization of a single-phase fluid, **not** a
quantitatively validated gas or spray phase. Rendering radii are sub-grid
representations, not measured bubble sizes. Record the model settings and
sources when presenting any additional case.

## Tests

Portable tests run with the foam conversion dependencies:

```bash
uv run --no-project --python 3.9 \
  --with meshio==5.3.5 --with numpy==2.0.2 --with partio==1.0.0 \
  python -m unittest discover -s tools/rendering/tests -v
```

The built FoamGenerator enables a seeded integration check; Blender 5.2
enables small liquid/foam/spray/bubble and stress renders with translated
bounds. Both tests are optional when those executables are unavailable:

```bash
TRIXIPARTICLES_FOAM_GENERATOR=/path/to/FoamGenerator \
TRIXIPARTICLES_BLENDER=/path/to/blender \
uv run --no-project --python 3.9 \
  --with meshio==5.3.5 --with numpy==2.0.2 --with partio==1.0.0 \
  python -m unittest discover -s tools/rendering/tests -v
```

The follow-up v03 migration PR will pin its exact CLI invocation, compare all
601 foam/spray/bubble source counts and particle hashes, inspect rendered
frames including the foam peak and stress peak, and retain the accepted v03
package until equivalence is verified. The current v03 media assets and
press-specific scripts are intentionally untouched by this tools PR.
