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
@inline function extract_svector(A, ::Val{NDIMS}, i...) where {NDIMS}
    # Explicit bounds check, which can be removed by calling this function with `@inbounds`
    @boundscheck checkbounds(A, NDIMS, i...)

    # Assume inbounds access now
    return SVector(ntuple(@inline(dim->@inbounds A[dim, i...]), NDIMS))
end

# Return `A[:, :, i]` as an `SMatrix`.
@propagate_inbounds function extract_smatrix(A, system, particle)
    return extract_smatrix(A, Val(ndims(system)), particle)
end

@inline function extract_smatrix(A, ::Val{NDIMS}, particle) where {NDIMS}
    @boundscheck checkbounds(A, NDIMS, NDIMS, particle)

    # Extract the matrix elements for this particle as a tuple to pass to SMatrix
    return SMatrix{NDIMS, NDIMS}(ntuple(@inline(i->@inbounds A[mod(i - 1, NDIMS) + 1,
                                                               div(i - 1, NDIMS) + 1,
                                                               particle]),
                                        Val(NDIMS^2)))
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

# Whether `system` can skip neighbors of `neighbor_system` at the same position.
# By default, this only depends on the kernel gradient of `system`.
# This can be dispatched if the kernel gradient of `neighbor_system` is also evaluated.
@inline skip_zero_distance(system, neighbor_system) = skip_zero_distance(system)

# Robust/safe version of the function below. In performance-critical code, manually check
# the kernel support, call `skip_zero_distance` and then `smoothing_kernel_grad_unsafe`.
@inline function smoothing_kernel_grad(system, pos_diff, distance, particle)
    return corrected_kernel_grad(system_smoothing_kernel(system), pos_diff, distance,
                                 smoothing_length(system, particle),
                                 system_correction(system), system, particle)
end

# Skip the zero distance and compact support checks for maximum performance.
# Only call this when you are sure that `0 < distance < compact_support`.
@inline function smoothing_kernel_grad_unsafe(system, pos_diff, distance, particle)
    return corrected_kernel_grad_unsafe(system_smoothing_kernel(system), pos_diff,
                                        distance, smoothing_length(system, particle),
                                        system_correction(system), system, particle)
end

# Kernel gradient of `system` for an interaction with `neighbor_system`.
# Particles at the same position are skipped if `skip_zero_distance(system, neighbor_system)`.
# If the correction of `neighbor_system` requires such pairs to be processed
# (e.g., because its kernel gradient does not vanish there), even though the kernel gradient
# of `system` vanishes, use the safe gradient, which handles particles at the same position.
@inline function interaction_kernel_grad(system, neighbor_system, pos_diff, distance,
                                         particle)
    if skip_zero_distance(system) && !skip_zero_distance(system, neighbor_system)
        return smoothing_kernel_grad(system, pos_diff, distance, particle)
    end
    return smoothing_kernel_grad_unsafe(system, pos_diff, distance, particle)
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
