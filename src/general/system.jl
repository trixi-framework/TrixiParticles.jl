# Abstract supertype for all system types.
abstract type System{NDIMS} end

abstract type FluidSystem{NDIMS} <: System{NDIMS} end
timer_name(::FluidSystem) = "fluid"
vtkname(system::FluidSystem) = "fluid"

abstract type SolidSystem{NDIMS} <: System{NDIMS} end
timer_name(::SolidSystem) = "solid"
vtkname(system::SolidSystem) = "solid"

abstract type BoundarySystem{NDIMS} <: System{NDIMS} end
timer_name(::BoundarySystem) = "boundary"
vtkname(system::BoundarySystem) = "boundary"

@inline function set_zero!(du)
    du .= zero(eltype(du))

    return du
end

initialize!(system, semi) = system

@inline Base.ndims(::System{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(system::System) = error("eltype not implemented for system $system")

# Number of integrated variables in the first component of the ODE system (coordinates)
@inline u_nvariables(system) = ndims(system)

# Number of integrated variables in the second component
# of the ODE system (velocity and sometimes density)
@inline v_nvariables(system) = ndims(system)

# Number of particles in the system
@inline nparticles(system) = length(system.mass)

# Number of particles in the system whose positions are to be integrated (corresponds to the size of u and du)
@inline n_moving_particles(system) = nparticles(system)

@inline eachparticle(system) = Base.OneTo(nparticles(system))

# Wrapper for systems with `SystemBuffer`
@inline each_moving_particle(system) = each_moving_particle(system, system.buffer)
@inline each_moving_particle(system, ::Nothing) = Base.OneTo(n_moving_particles(system))

@inline active_coordinates(u, system) = active_coordinates(u, system, system.buffer)
@inline active_coordinates(u, system, ::Nothing) = current_coordinates(u, system)

@inline active_particles(system) = active_particles(system, system.buffer)
@inline active_particles(system, ::Nothing) = eachparticle(system)

# This should not be dispatched by system type. We always expect to get a column of `A`.
@propagate_inbounds function extract_svector(A, system, i)
    extract_svector(A, Val(ndims(system)), i)
end

# Return the `i`-th column of the array `A` as an `SVector`.
@inline function extract_svector(A, ::Val{NDIMS}, i) where {NDIMS}
    # Explicit bounds check, which can be removed by calling this function with `@inbounds`
    @boundscheck checkbounds(A, NDIMS, i)

    # Assume inbounds access now
    return SVector(ntuple(@inline(dim->@inbounds A[dim, i]), NDIMS))
end

# Return `A[:, :, i]` as an `SMatrix`.
@inline function extract_smatrix(A, system, particle)
    # Extract the matrix elements for this particle as a tuple to pass to SMatrix
    return SMatrix{ndims(system),
                   ndims(system)}(ntuple(@inline(i->A[mod(i - 1, ndims(system)) + 1,
                                                      div(i - 1, ndims(system)) + 1,
                                                      particle]),
                                         Val(ndims(system)^2)))
end

# Specifically get the current coordinates of a particle for all system types.
@propagate_inbounds function current_coords(u, system, particle)
    return extract_svector(current_coordinates(u, system), system, particle)
end

# This can be dispatched by system type, since for some systems, the current coordinates
# are stored in u, for others in the system itself. By default, try to extract them from u.
@inline current_coordinates(u, system) = u

# Specifically get the initial coordinates of a particle for all system types
@inline function initial_coords(system, particle)
    return extract_svector(initial_coordinates(system), system, particle)
end

# This can be dispatched by system type
@inline initial_coordinates(system) = system.initial_condition.coordinates

@propagate_inbounds function current_velocity(v, system, particle)
    return extract_svector(current_velocity(v, system), system, particle)
end

# This can be dispatched by system type, since for some systems, the current velocity
# is stored in `v`, for others it might be stored elsewhere.
# By default, try to extract it from `v`.
@inline current_velocity(v, system) = v

@inline function current_density(v, system::System, particle)
    return current_density(v, system)[particle]
end

@propagate_inbounds function current_pressure(v, system::System, particle)
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

@inline function smoothing_kernel_grad(system, pos_diff, distance, particle)
    return corrected_kernel_grad(system_smoothing_kernel(system), pos_diff,
                                 distance, smoothing_length(system, particle),
                                 system_correction(system), system, particle)
end

# System update orders. This can be dispatched if needed.
function update_positions!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_quantities!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_pressure!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_boundary_interpolation!(system, v, u, v_ode, u_ode, semi, t;
                                        update_from_callback=false)
    return system
end

function update_final!(system, v, u, v_ode, u_ode, semi, t; update_from_callback=false)
    return system
end

# Only for systems requiring a mandatory callback
reset_callback_flag!(system) = system

initial_smoothing_length(system) = smoothing_length(system, nothing)

function smoothing_length(system, particle)
    return system.smoothing_length
end

system_smoothing_kernel(system) = system.smoothing_kernel
system_correction(system) = nothing

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
