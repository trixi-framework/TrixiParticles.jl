# Accepted C01 v03 media with the CLI rendering tools

This command-line handoff uses the audited 601-state, 300 Hz C01 v38 converted
VTK, reconstruction, whitewater cache, and stress-surface export. Existing
surface and whitewater PLYs are **already in Blender `xzy` coordinates**;
another axis swap corrupts them. JSON metadata is a generated data inventory,
not a case configuration file. Set the paths for your copy of those inputs and
for **new** outputs:

```bash
PIPELINE=tools/rendering/pipeline.py
BLENDER=/snap/bin/blender
GENERATOR=/path/to/patched/SPlisHSPlasH/bin/FoamGenerator
FLUID_PVD=/path/to/v03_render_input/vtk/fluid_1.pvd
SURFACE_DIR=/path/to/v03_surface
FOAM_DIR=/path/to/v03_accepted_foam
STRESS_DIR=/path/to/v03_stress_surface_peak
OUT=/path/to/new/v03_validation
```

The reconstruction's `blade_files` and accepted foam's `particle_counts` and
`filename` fields are supported directly. The renderer checks the old foam's
run and manifest hashes against the reconstruction and validates selected PLY
hashes. It does not modify or copy the approved cache. New foam generation
produces the new tools' metadata schema instead.

## Regenerate whitewater into a new directory

Use all 601 frames in order: automatic calibration needs the entire sequence,
and marker history depends on earlier frames. These values come from the
accepted `foam_metadata.json`, including the Float32 spacing-derived radius
and actual discretized tank bounds:

```bash
uv run --no-project --python 3.9 \
  --with meshio==5.3.5 --with numpy==2.0.2 --with partio==1.0.0 \
  python "$PIPELINE" foam \
  --fluid-pvd "$FLUID_PVD" --generator "$GENERATOR" \
  --generator-source /path/to/patched/SPlisHSPlasH/Tools/FoamGenerator/main.cpp \
  --generator-patch tools/rendering/build/splishsplash_foam_generator.patch \
  --output "$OUT/foam_generated" \
  --domain-min 0 0 0 \
  --domain-max 1.996649980545044 4.000185012817383 0.998324990272522 \
  --generator-radius 0.0034425000194460154 --foam-scale 1000 \
  --lifetime-min 2 --lifetime-max 5 --buoyancy 2 \
  --drag 0.8 --drag-reference-step 0.02 \
  --seed 20260915 --generator-threads 1 --velocity-array velocity \
  --output-axis-order xzy
```

Build `GENERATOR` using [the pinned build instructions](README.md#prerequisites).
The old binary hardcoded seed 20260915, while the new patched binary accepts
`--seed`. Equal settings alone do **not** establish equivalent particles:
compare all 601 per-type raw, clipped, and packaged counts and each PLY's
position/velocity payload against the accepted cache. The new PLY header
describes its coordinate order differently, so whole-file SHA-256 differs even
when payloads agree. Preserve the approved cache until these checks pass.

## Accepted clean and whitewater scenes

The following Bash array holds only explicit CLI flags. It records the old
horizontal cutaway's pedestal, glass bevel/overhang, four lights, color
management, and independently selected material roles.

```bash
scene=(
  --blender "$BLENDER" --surface-dir "$SURFACE_DIR"
  --surface-metadata "$SURFACE_DIR/reconstruction_metadata.json"
  --surface-axis-order xzy --foam-axis-order xzy
  --camera-position 3.15 2.35 1.45 --camera-target 1 0.5 0.34 --camera-fov 34
  --tank-min 0 0 0
  --tank-max 1.996649980545044 4.000185012817383 0.998324990272522
  --light 'key:0.2:2.8:3.8:1250:1:0.76:0.58:3.2'
  --light 'fill:3.7:2.2:1.9:850:0.34:0.64:1:2.6'
  --light 'rim:-1.3:-1:2.5:1100:0.20:0.75:1:2'
  --light 'top:1:0.4:4.5:250:1:0.92:0.78:2.2'
  --width 3840 --height 2160 --samples 96 --engine cycles
  --device gpu --gpu-backend OPTIX --max-bounces 10
  --transmission-bounces 8 --transparent-bounces 8
  --view-transform AgX --look 'AgX - Medium High Contrast' --exposure 0.7
  --world-color 0.002 0.005 0.012 --world-strength 0.16
  --solid-floor-extension 0.02 --solid-bevel 0.005
  --floor-material dark-metal --pedestal-margin 0.14
  --pedestal-thickness 0.08 --pedestal-center-offset -0.055 --floor-bevel 0.025
  --glass-material low-iron-glass --wall-thickness 0.018
  --floor-thickness 0.018 --visible-wall-height 0.16
  --wall-cap-extension 0.02 --glass-overhang 0.02 --glass-bevel 0.006
  --liquid-material turbulent-water --solid-material anodized-copper
)

python3 "$PIPELINE" render "${scene[@]}" \
  --frame 174 --output "$OUT/clean_174.png"

python3 "$PIPELINE" render "${scene[@]}" \
  --foam-dir "$FOAM_DIR" --foam-material whitewater-froth \
  --spray-material water-droplet --bubble-material submerged-air \
  --frame 174 --output "$OUT/foam_174.png"
```

Select `--foam-dir "$OUT/foam_generated"` after its counts and payloads are
verified. Use `--sequence --start 0 --stop 601 --output "$OUT/clean_frames"`
(or `foam_frames`) to render a new movie. For lower-cost previews, change
`--width`, `--height`, `--samples`, and `--device` entries in `scene` before
running a still. `--resume` requires an interrupted sequence with unchanged
input hashes and scene settings.

## Stress close-up, source timestep 179

The audited stress export's own `water_context.ply` and five pre-colored blade
PLYs are render inputs; reconstruction metadata supplies the time/frame map.
This close-up has no glass walls.

```bash
python3 "$PIPELINE" render \
  --blender "$BLENDER" --surface-dir "$SURFACE_DIR" \
  --surface-metadata "$SURFACE_DIR/reconstruction_metadata.json" \
  --mode stress --frame 179 --surface-axis-order xzy --foam-axis-order xzy \
  --stress-export-metadata "$STRESS_DIR/stress_surface_metadata.json" \
  --stress-water-mesh "$STRESS_DIR/water_context.ply" \
  --stress-mesh "$STRESS_DIR/blade_01_stress.ply" \
  --stress-mesh "$STRESS_DIR/blade_02_stress.ply" \
  --stress-mesh "$STRESS_DIR/blade_03_stress.ply" \
  --stress-mesh "$STRESS_DIR/blade_04_stress.ply" \
  --stress-mesh "$STRESS_DIR/blade_05_stress.ply" \
  --ghost-material v03-ghost-water --stress-normal-light 0.28 \
  --floor-material neutral-stress --solid-floor-extension 0.02 \
  --solid-bevel 0.004 --pedestal-margin 0.09 \
  --pedestal-thickness 0.055 --pedestal-center-offset -0.032 \
  --floor-bevel 0.025 \
  --camera-position -0.55 -0.05 1.18 --camera-target 1.1 0.5 0.2 \
  --camera-fov 29 --tank-min 0 0 0 \
  --tank-max 1.996649980545044 4.000185012817383 0.998324990272522 \
  --light 'warm:2.4:-1.4:3:240:1:0.82:0.68:2.5' \
  --light 'fill:0:2.7:2.6:180:0.68:0.86:1:3.2' \
  --light 'top:1.1:0.5:3.5:140:1:0.97:0.90:2' \
  --light 'base:1:1.8:0.35:60:1:0.53:0.32:1.4' \
  --width 3840 --height 2160 --samples 96 --engine cycles \
  --device gpu --gpu-backend OPTIX --max-bounces 8 \
  --transmission-bounces 6 --transparent-bounces 8 \
  --view-transform Standard --look None --exposure 0 \
  --world-color 0.055 0.075 0.1 --world-strength 0.65 \
  --output "$OUT/stress_179.png"
```

The released v03 media and metadata remain the reference. Compare new
production-setting stills with the unbranded v03 PNGs, including the foam
peak and stress peak, before promoting a new sequence. Denoising on a
different GPU driver can change pixel hashes; record image comparisons and
settings rather than presuming bitwise identity.

## Validation on the accepted v03 release (2026-09-29)

- The audited `fluid_1.pvd` SHA-256 is
  `f99156c41013f560b0eb05e3291be02655e8169f2fcfe71b2ad205df2e587c43`;
  its 601 times have a 1/300-second effective interval. On source frame 0,
  both converters wrote the **same Partio SHA-256**
  `1c902d32fa7e37b5d5ec49da74669553c0d23ce1d4162c8b3998b3bb783a813d`
  for 1,097,505 fluid particles.
- A one-off local audit checked all 601 accepted whitewater records, category
  totals, clipping, PLY attributes and recorded file hashes. The foam peak is
  58,326 at frame 174.
- The new renderer read that cache without conversion, and produced GPU/OptiX
  3840×2160, 96-sample frame-174 stills. Against the released unbranded v03
  originals (whose hashes match the release render manifests), clean water
  has mean absolute RGB error **0.0814/255** with **91.94%** exactly matching
  pixels; whitewater has **0.0653/255** with **93.38%** exactly matching pixels.
  The 95th-percentile maximum channel error per pixel is 1 for both views.
- The stress frame-179 render at the same production settings has mean
  absolute RGB error **0.000029/255** and **99.991%** exactly matching pixels
  against its approved output, with the stress-export inventory and hashes
  checked. The stress export's water PLY is byte-for-byte the corresponding
  reconstruction frame.
- The new seeded FoamGenerator has passed the reusable tools' integration
  checks, but a **full 601-state regeneration and candidate-versus-reference
  payload comparison has not yet run**. Likewise these still checks do not
  validate 601 newly rendered PNGs. Keep accepted outputs and original press
  scripts as provenance until those full checks are complete.
