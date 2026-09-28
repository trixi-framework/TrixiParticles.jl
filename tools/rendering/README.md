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

## Quick start

Most cases need only a material preset plus scene geometry. Three presets
cover the reviewed v03 looks; any single flag can still override a preset
value (see [Material presets](#material-presets)).

Clean-water still from native PR #1329 output:

```bash
python tools/rendering/pipeline.py render \
  --blender /path/to/blender --surface-dir /path/to/surface \
  --surface-pvd /path/to/surface/surface_fluid_1.pvd \
  --solid-pattern 'surface_structure_1_{iter}.ply' \
  --output /path/to/new/still.png --frame 0 \
  --material-preset liquid \
  --surface-axis-order xyz --foam-axis-order xyz \
  --camera-position 3 2 2 --camera-target 1 0.5 0.3 --camera-fov 34 \
  --tank-min 0 0 0 --tank-max 2 4 1 \
  --light 'key:0.2:2.8:3.8:1250:1:0.76:0.58:3.2' \
  --width 640 --height 360 --samples 16 --device cpu \
  --world-color 0.002 0.005 0.012 --world-strength 0.16
```

Whitewater sequence (generate once, then render a frame range):

```bash
python tools/rendering/pipeline.py foam \
  --fluid-pvd /path/to/fluid.pvd \
  --generator /path/to/SPlisHSPlasH/bin/FoamGenerator \
  --output /path/to/new/whitewater \
  --domain-min 0 0 0 --domain-max 2 4 1 \
  --generator-radius 0.01 --foam-scale 1000 \
  --lifetime-min 2 --lifetime-max 5 --buoyancy 2 \
  --drag 0.8 --drag-reference-step 0.02 \
  --seed 144 --generator-threads 1 \
  --output-axis-order xyz

python tools/rendering/pipeline.py render \
  --blender /path/to/blender --surface-dir /path/to/surface \
  --surface-pvd /path/to/surface/surface_fluid_1.pvd \
  --solid-pattern 'surface_structure_1_{iter}.ply' \
  --foam-dir /path/to/new/whitewater \
  --output /path/to/new/frames --sequence --start 0 \
  --material-preset foam \
  --surface-axis-order xyz --foam-axis-order xyz \
  --camera-position 3 2 2 --camera-target 1 0.5 0.3 --camera-fov 34 \
  --tank-min 0 0 0 --tank-max 2 4 1 \
  --light 'key:0.2:2.8:3.8:1250:1:0.76:0.58:3.2' \
  --width 640 --height 360 --samples 16 \
  --world-color 0.002 0.005 0.012 --world-strength 0.16
```

Stress still from a pre-colored PLY:

```bash
python tools/rendering/pipeline.py render \
  --blender /path/to/blender --surface-dir /path/to/surface \
  --surface-pvd /path/to/surface/surface_fluid_1.pvd \
  --output /path/to/new/stress.png --frame 179 \
  --mode stress --stress-mesh /path/to/stress_blade_01.ply \
  --stress-mesh /path/to/stress_blade_02.ply \
  --material-preset stress \
  --surface-axis-order xyz --foam-axis-order xyz \
  --camera-position 3 2 2 --camera-target 1 0.5 0.3 --camera-fov 34 \
  --tank-min 0 0 0 --tank-max 2 4 1 \
  --light 'key:0.2:2.8:3.8:1250:1:0.76:0.58:3.2' \
  --width 640 --height 360 --samples 16 \
  --world-color 0.002 0.005 0.012 --world-strength 0.16
```

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
`uv run --with cmake --with ninja`). Re-running the script against the same
directories is idempotent. The executable appears at
`/path/to/SPlisHSPlasH/bin/FoamGenerator`. Its `--help` output must include
`--seed`; the wrapper rejects older binaries.

## 1. Generate whitewater

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

Option reference:

| Flag | Meaning |
|---|---|
| `--fluid-pvd` | Source fluid VTK collection; every frame is read in order |
| `--generator` | Patched FoamGenerator executable (must accept `--seed`) |
| `--generator-source`, `--generator-patch` | Optional fingerprints recorded in the output manifest |
| `--output` | New or empty directory for `frame_*.ply` plus `foam_metadata.json` |
| `--domain-min`, `--domain-max` | Kill-box bounds; particles outside are clipped and counted |
| `--generator-radius` | Smoothing length used by the generator |
| `--foam-scale` | Global secondary-particle multiplier |
| `--lifetime-min`, `--lifetime-max` | Advected particle lifetime range in seconds |
| `--buoyancy` | Buoyancy coefficient |
| `--drag`, `--drag-reference-step` | Drag at the reference step; converted to the source cadence |
| `--seed` | Required 64-bit deterministic seed |
| `--generator-threads` | `OMP_NUM_THREADS` for the generator run |
| `--velocity-array` | Name of the source velocity point-data array |
| `--output-axis-order` | `xyz` or `xzy` coordinate space of the packaged PLY |
| `--reuse` | Reuse only a complete, hash-validated cache with identical provenance |

The generator always reads the **complete specified PVD sequence** in physical
order before producing the requested output; its automatic potential limits
are calibrated on that complete sequence. Choose a render frame in stage 2,
not by starting the generator in the middle of an evolving event.

`foam_metadata.json` records the source timestamps/hashes, generator binary,
seed, calibrated log, per-type and clipped counts, the effective drag after
cadence conversion, and output hashes. A complete cache may be reused with the
same command plus `--reuse`; it validates inputs and every output. Incomplete
generation must be restarted into a fresh directory, since resetting particle
history at a later frame changes the model. The output PLY contains position
and velocity in the specified axis order, even when only one of the three
secondary types is present.

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

Option reference:

| Group | Flags |
|---|---|
| Inputs | `--blender`, `--surface-dir`, `--surface-pvd` or `--surface-metadata`, `--foam-dir`, `--solid-pattern`, `--surface-axis-order`, `--foam-axis-order` |
| Selection | `--frame N` for one still, or `--sequence` with `--start/--stop/--stride`; `--resume` continues a validated prefix |
| Scene | `--camera-position`, `--camera-target`, `--camera-fov`, `--tank-min`, `--tank-max`, `--light` (repeatable `name:x:y:z:energy:r:g:b:size`), `--world-color`, `--world-strength` |
| Image | `--width`, `--height`, `--samples`, `--engine {cycles,eevee}`, `--device {cpu,gpu}`, `--gpu-backend`, `--view-transform`, `--look`, `--exposure`, `--gamma`, `--max-bounces`, `--transmission-bounces`, `--transparent-bounces`, `--png-compression`, `--no-denoise` |
| Materials | `--material-preset {liquid,foam,stress}` plus the individual `--water-*`, `--blade-*`, `--floor-*`, `--glass-*`, `--wall-*`, `--foam-*`, `--spray-*`, `--bubble-*` flags below |
| Stress | `--mode stress --stress-mesh FILE` (repeat for each solid), `--stress-water-opacity`, `--stress-water-transmission`, `--stress-water-specular`, `--stress-normal-light`, `--stress-light-direction` |

`render --sequence` writes numbered PNGs plus `render_metadata.json` and
`render_progress.json`. `--resume` validates the settings, source hashes,
renderer source, and PNG prefix before continuing. An already complete
sequence cannot be resumed or silently overwritten.

For pre-colored stress PLYs, repeat `--stress-mesh FILE` for each solid;
the v03 stress view uses **five** blades. Optionally set
`--stress-water-opacity` and `--stress-normal-light`. The v03 stress image uses
an independently color-encoded YlOrRd PLY; this tool preserves the input
vertex colors, its stress emission material, and the contextual ghost-water
and floor treatments. Stress export itself remains an upstream input. The
`stress` preset supplies the reviewed `0.005` ghost-water opacity, its distinct
color/transmission/specular response, and the `0.72–1.0` world-normal luminance
factor. Without a preset, `--stress-water-opacity 0` omits contextual water;
otherwise pass `--water-color`, `--water-roughness`, `--water-ior`,
`--stress-water-transmission`, and `--stress-water-specular` explicitly. The
ghosted shader is separate from the primary liquid-volume shader.

## Material presets

Only three material sets exist. A preset fills every material flag it covers;
any flag passed explicitly wins over the preset. Flags outside all presets
(camera, lights, tank, resolution) are always required individually. Run
`python tools/rendering/pipeline.py material-presets` to print every preset
value.

| Preset | Contents | Use with |
|---|---|---|
| `liquid` | Physical water (Pope–Fry absorption, scattering), copper solids, glass tank, floor, tank construction | Clean surface stills/sequences |
| `foam` | The `liquid` set plus froth, spray, and bubble materials | `--foam-dir` whitewater renders |
| `stress` | YlOrRd vertex-color emission, `0.72–1.0` normal modulation, copper-edge/floor and `0.005` ghost water with its own physical material | `--mode stress --stress-mesh` |

Override example (scene settings remain CLI inputs):

```bash
python tools/rendering/pipeline.py render \
  --blender /path/to/blender --surface-dir /path/to/surface \
  --surface-pvd /path/to/surface/surface_fluid_1.pvd \
  --foam-dir /path/to/whitewater --output /path/to/new/still.png --frame 0 \
  --material-preset foam --foam-point-radius 0.008 \
  --surface-axis-order xyz --foam-axis-order xyz \
  --camera-position 3 2 2 --camera-target 1 0.5 0.3 --camera-fov 34 \
  --tank-min 0 0 0 --tank-max 2 4 1 \
  --light 'key:0.2:2.8:3.8:1250:1:0.76:0.58:3.2' \
  --width 640 --height 360 --samples 16 \
  --world-color 0.002 0.005 0.012 --world-strength 0.16
```

The presets encode v03's **material-node parameters**, not a complete v03
scene. `liquid` and `foam` share the same physical water material, with
absorption `(0.85, 0.14125, 0.02305) m⁻¹`, scattering `(0.02, 0.04, 0.08) m⁻¹`,
and anisotropy `0.35`. The accepted tank base uses `(0.012, 0.018, 0.028)`
linear RGB; the glass pane has transmission `0.90` and specular IOR level
`0.04`. `stress` instead uses its `(0.020, 0.032, 0.050)` pedestal,
`0.28`-weighted normal modulation, and ghost water with color
`(0.015, 0.28, 0.46)`, transmission `0.72`, and specular level `0.18`.
The bubble relative IOR is exactly `1 / 1.333`. These values are checked
against the accepted v03 render metadata and source shaders. Explicit flags
still override any one value.

**Can the presets be used for v03 reproduction?** They supply the reviewed
material parameters, yes. A matching v03 render additionally needs the
original camera/lights, tank and pedestal geometry, frame selection,
coordinate mapping, Blender/device/color management, accepted reconstruction
and foam caches, and any render-only blade-floor treatment. Those settings
are *not* preset; the follow-up v03 migration PR will pass them explicitly
and establish full render and source-frame parity. This tools PR does not
claim bitwise v03 image reproduction. The rejected `legacy-cyan` and
`physical-clear` water branches and the redundant `foaming-water` preset
are not implemented.

## Materials and scientific scope

There is **one** physically based transmitted/scattered liquid shader, with
all coefficients supplied by CLI flags or presets. Optional foam,
velocity-aligned spray, and bubble interfaces use their own parameters.
Solids, glass, the floor, and the stress view retain configurable materials.
Enabling whitewater changes the scene, not the liquid material. Camera,
lights, render engine/device, color management, and all render-only geometry
offsets are explicit flags.

Whitewater is a heuristic visualization of a single-phase fluid, **not** a
quantitatively validated gas or spray phase. Rendering radii are sub-grid
representations, not measured bubble sizes. Record the model settings and
sources when presenting any additional case.

## Troubleshooting

- `Blender executable not found`: pass the real binary; a snap launcher path
  must be given unresolved (the wrapper does not resolve it).
- `FoamGenerator must support the explicit --seed build patch`: rebuild with
  `build/build_foam_generator.py`; unpatched upstream binaries are rejected.
- `source fluid frames must be uniformly spaced`: the foam stage needs a
  constant cadence; check the PVD timestamps.
- `whitewater cache provenance or frame count differs`: inputs or settings
  changed; regenerate into a fresh directory instead of reusing.
- `resume input, settings or renderer do not match`: resume only continues an
  interrupted run with identical inputs, flags, and worker source.
- Missing CUDA/OptiX devices abort GPU renders explicitly instead of silently
  falling back to CPU.

## Tests

Portable tests run with the foam conversion dependencies:

```bash
uv run --no-project --python 3.9 \
  --with meshio==5.3.5 --with numpy==2.0.2 --with partio==1.0.0 \
  python -m unittest discover -s tools/rendering/tests -v
```

The built FoamGenerator enables a seeded integration check; Blender 5.2
enables small liquid/foam/spray/bubble and stress renders with translated
bounds, including a preset-based render. Both tests are optional when those
executables are unavailable:

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
