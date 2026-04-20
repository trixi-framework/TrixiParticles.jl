# Abstract supertype for all system types.
abstract type AbstractSystem{NDIMS} end

@inline Base.ndims(::AbstractSystem{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(system::AbstractSystem) = error("eltype not implemented for system $system")

abstract type AbstractFluidSystem{NDIMS} <: AbstractSystem{NDIMS} end
timer_name(::AbstractFluidSystem) = "fluid"
vtkname(system::AbstractFluidSystem) = "fluid"

abstract type AbstractStructureSystem{NDIMS} <: AbstractSystem{NDIMS} end
timer_name(::AbstractStructureSystem) = "structure"
vtkname(system::AbstractStructureSystem) = "structure"

abstract type AbstractBoundarySystem{NDIMS} <: AbstractSystem{NDIMS} end
timer_name(::AbstractBoundarySystem) = "boundary"
vtkname(system::AbstractBoundarySystem) = "boundary"

# Number of integrated variables in the first component of the ODE system (coordinates)
@inline u_nvariables(system) = ndims(system)

# Number of integrated variables in the second component
# of the ODE system (velocity and sometimes density)
@inline v_nvariables(system) = ndims(system)

# Number of particles in the system
@inline nparticles(system) = length(system.mass)

# Number of particles in the system whose positions are to be integrated (corresponds to the size of u and du)
@inline n_integrated_particles(system) = nparticles(system)

@inline eachparticle(system::AbstractSystem) = each_active_particle(system)
@inline eachparticle(initial_condition) = Base.OneTo(nparticles(initial_condition))

# Wrapper for systems with `SystemBuffer`
@inline each_integrated_particle(system) = each_integrated_particle(system, buffer(system))
@inline function each_integrated_particle(system, ::Nothing)
    return Base.OneTo(n_integrated_particles(system))
end

@inline active_coordinates(u, system) = active_coordinates(u, system, buffer(system))
@inline active_coordinates(u, system, ::Nothing) = current_coordinates(u, system)

@inline each_active_particle(system) = each_active_particle(system, buffer(system))
@inline each_active_particle(system, ::Nothing) = Base.OneTo(nparticles(system))

@inline function set_zero!(du)
    du .= zero(eltype(du))

    return du
end

initialize!(system, semi) = system

# This should not be dispatched by system type. We always expect the first index of `A`
# to enumerate spatial dimensions.
@propagate_inbounds function extract_svector(A, system, i...)
    extract_svector(A, Val(ndims(system)), i...)
end

# Return `A[:, i...]` as an `SVector`.
@inline function extract_svector(A, ::Val{NDIMS}, i) where {NDIMS}
    # Explicit bounds check, which can be removed by calling this function with `@inbounds`
    @boundscheck checkbounds(A, NDIMS, i...)

    # Assume inbounds access now
    return SVector(ntuple(@inline(dim->@inbounds A[dim, i...]), NDIMS))
    # vec = SIMD.vload(SIMD.Vec{NDIMS, eltype(A)}, pointer(A, NDIMS * (i - 1) + 1))

    return SVector{NDIMS}(Tuple(vec))
end

# Return `A[:, :, i]` as an `SMatrix`.
@propagate_inbounds function extract_smatrix(A, system, i)
    return extract_smatrix(A, Val(ndims(system)), i)
end

@inline function extract_smatrix(A::AbstractArray{T, 3}, ::Val{N}, i) where {T, N}
    @boundscheck checkbounds(A, N, N, i)
    # This function assumes that the first two dimensions of `A` have exactly the size `N`.
    @boundscheck if stride(A, 3) != N^2
        # WARNING: Don't split this string with `*`, or this function won't compile on GPUs,
        # even when the error is never thrown.
        error("extract_smatrix only works for 3D arrays where the first two dimensions each have size N")
    end

    # Extract the matrix elements as a tuple in column-major order,
    # and construct an `SMatrix` from it.
    return SMatrix{N, N}(ntuple(@inline(j->@inbounds A[(i - 1) * N^2 + j]), Val(N^2)))
end

@inline function extract_smatrix(A::AbstractArray{T, 3}, ::Val{2}, i) where {T}
    @boundscheck checkbounds(A, 2, 2, i)
    # This function assumes that the first two dimensions of `A` have exactly the size `2`.
    @boundscheck if stride(A, 3) != 4
        # WARNING: Don't split this string with `*`, or this function won't compile on GPUs,
        # even when the error is never thrown.
        error("extract_smatrix only works for 3D arrays where the first two dimensions each have size N")
    end

    # This is the same as
    # `SMatrix{N, N}(ntuple(@inline(j->@inbounds A[(i - 1) * N^2 + j]), Val(N^2)))`,
    # but it's slightly faster on CPUs in some cases. As opposed to the GPU-optimized
    # version in `extract_smatrix_aligned`, we use `vload` and not `vloada` here, which
    # does not require alignment and works if N^2 is not a power of 2.
    # This is faster in the TLSPH RHS in 2D, but slower in 3D for some reason.
    # See the benchmarks in https://github.com/trixi-framework/TrixiParticles.jl/pull/1147.
    vec = SIMD.vload(SIMD.Vec{4, T}, pointer(A, 4 * (i - 1) + 1))
    return SMatrix{2, 2}(Tuple(vec))
end

# Specifically get the current coordinates of a particle for all system types.
@propagate_inbounds function current_coords(u, system, particle)
    return extract_svector(current_coordinates(u, system), system, particle)
end

# This can be dispatched by system type, since for some systems, the current coordinates
# are stored in u, for others in the system itself. By default, try to extract them from u.
@inline current_coordinates(u, system) = u

# Specifically get the initial coordinates of a particle for all system types
@propagate_inbounds function initial_coords(system, particle)
    return extract_svector(initial_coordinates(system), system, particle)
end

# This can be dispatched by system type
@inline initial_coordinates(system) = system.initial_condition.coordinates

@inline coordinates_eltype(system::AbstractSystem) = eltype(initial_coordinates(system))

@propagate_inbounds function current_velocity(v, system, particle)
    return extract_svector(current_velocity(v, system), system, particle)
end

# This can be dispatched by system type, since for some systems, the current velocity
# is stored in `v`, for others it might be stored elsewhere.
# By default, try to extract it from `v`.
@inline current_velocity(v, system) = v

@propagate_inbounds function current_density(v, system::AbstractSystem, particle)
    return current_density(v, system)[particle]
end

@propagate_inbounds function current_pressure(v, system::AbstractSystem, particle)
    return current_pressure(v, system)[particle]
end

@inline function current_acceleration(system, particle)
    # TODO: Return `dv` of solid particles
    return zero(SVector{ndims(system), eltype(system)})
end

@inline set_particle_density!(v, system, particle, density) = v
@inline set_particle_pressure!(v, system, particle, pressure) = v

@inline function smoothing_kernel(system, distance, particle)
    (; smoothing_kernel) = system
    return kernel(smoothing_kernel, distance, smoothing_length(system, particle))
end

@inline function smoothing_kernel_unsafe(system, distance, particle)
    (; smoothing_kernel) = system
    return kernel_unsafe(smoothing_kernel, distance, smoothing_length(system, particle))
end

@inline function skip_zero_distance(system::AbstractSystem)
    return skip_zero_distance(system_correction(system))
end

# Robust/safe version of the function below. In performance-critical code, manually check
# the kernel support, call `skip_zero_distance` and then `smoothing_kernel_grad_unsafe`.
@inline function smoothing_kernel_grad(system, pos_diff, distance, particle)
    h = smoothing_length(system, particle)
    compact_support_ = compact_support(system_smoothing_kernel(system), h)

    # Note that `sqrt(eps(h^2)) != eps(h)`
    if distance >= compact_support_ ||
       (skip_zero_distance(system) && distance^2 < eps(h^2))
        return zero(pos_diff)
    end

    return smoothing_kernel_grad_unsafe(system, pos_diff, distance, particle)
end

# Skip the zero distance and compact support checks for maximum performance.
# Only call this when you are sure that `0 < distance < compact_support`.
@inline function smoothing_kernel_grad_unsafe(system, pos_diff, distance, particle)
    return corrected_kernel_grad_unsafe(system_smoothing_kernel(system), pos_diff,
                                        distance, smoothing_length(system, particle),
                                        system_correction(system), system, particle)
end

# System updates do nothing by default, but can be dispatched if needed
function update_positions!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_quantities!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_pressure!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_boundary_interpolation!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_final!(system, v, u, v_ode, u_ode, semi, t; kwargs...)
    return system
end

function reset_interaction_caches!(system)
    return system
end

function finalize_interaction!(system, dv, v, u, dv_ode, v_ode, u_ode, semi)
    return system
end

@inline initial_smoothing_length(system) = smoothing_length(system, nothing)

@inline function smoothing_length(system, particle)
    return system.smoothing_length
end

@inline system_smoothing_kernel(system) = system.smoothing_kernel
@inline system_correction(system) = nothing

@inline particle_spacing(system, particle) = system.initial_condition.particle_spacing

# Assuming a constant particle spacing one can calculate the number of neighbors within the
# compact support for an undisturbed particle distribution.
function ideal_neighbor_count(::Val{D}, particle_spacing, compact_support) where {D}
    throw(ArgumentError("Unsupported dimension: $D"))
end

@inline function ideal_neighbor_count(::Val{2}, particle_spacing, compact_support)
    return floor(Int, pi * compact_support^2 / particle_spacing^2)
end

@inline @fastpow function ideal_neighbor_count(::Val{3}, particle_spacing, compact_support)
    return floor(Int, 4 // 3 * pi * compact_support^3 / particle_spacing^3)
end
