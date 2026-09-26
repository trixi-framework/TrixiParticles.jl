# Production configuration: constants and the reconstruction options struct.
const Face = SVector{3, Int32}
const MESH_CLEANUP_TOLERANCE_M = 1.0e-6
const GAUSSIAN_TRUNCATE = 4.0
const SPARSE_COMPONENT_GRAPH_RADIUS_PER_SPACING = 1.5
const SPARSE_COMPONENT_MAXIMUM_PARTICLES = 64
const SPARSE_FALLBACK_MAXIMUM_ITERATIONS = 4
const SPARSE_FALLBACK_LATITUDE_SEGMENTS = 16
const SPARSE_FALLBACK_LONGITUDE_SEGMENTS = 32
# Deterministic parallel reductions (mesh statistics) accumulate into this many buckets
# covering contiguous face ranges. The count is fixed, so parallel execution agrees
# bitwise across thread counts; one bucket reproduces the exact serial summation order.
const PARALLEL_REDUCTION_BUCKETS = 64
# Meshes with fewer faces than this always use the serial reductions, where threading
# overhead would exceed the gain (and which keeps small-mesh results bitwise identical
# across backends)
const PARALLEL_REDUCTION_MIN_FACES = 40000

Base.@kwdef struct SurfaceReconstructionOptions
    isovalue::Float64 = 0.5
    volume_tolerance_percent::Float64 = 0.1
    volume_max_iterations::Int = 8
    minimum_isovalue::Float64 = 0.1
    maximum_isovalue::Float64 = 0.9
    warm_start::Bool = true
    record_field_moments::Bool = false
    sparse_component_fallback::Bool = false
    keep_enclosed_fluid::Bool = false
end
