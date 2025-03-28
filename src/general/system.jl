# Abstract supertype for all system types. We additionally store the type of the system's
# initial condition, which is `Nothing` when using KernelAbstractions.jl.
abstract type System{NDIMS, IC} end

# When using KernelAbstractions.jl, the initial condition has been replaced by `nothing`
GPUSystem = System{NDIMS, Nothing} where {NDIMS}

abstract type FluidSystem{NDIMS, IC} <: System{NDIMS, IC} end
timer_name(::FluidSystem) = "fluid"
vtkname(system::FluidSystem) = "fluid"

abstract type SolidSystem{NDIMS, IC} <: System{NDIMS, IC} end
timer_name(::SolidSystem) = "solid"
vtkname(system::SolidSystem) = "solid"

abstract type BoundarySystem{NDIMS, IC} <: System{NDIMS, IC} end
timer_name(::BoundarySystem) = "boundary"
vtkname(system::BoundarySystem) = "boundary"

@inline function set_zero!(du)
    du .= zero(eltype(du))

    return du
end

initialize!(system, neighborhood_search) = system

@inline Base.ndims(::System{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(system::System) = eltype(system.initial_condition)

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

# Specifically get the initial coordinates of a particle for all system types.
@inline function initial_coords(system, particle)
    return extract_svector(initial_coordinates(system), system, particle)
end

# This can be dispatched by system type.
@inline initial_coordinates(system) = system.initial_condition.coordinates

@propagate_inbounds current_velocity(v, system, particle) = extract_svector(v, system,
                                                                            particle)

@inline function current_acceleration(system, particle)
    # TODO: Return `dv` of solid particles
    return zero(SVector{ndims(system), eltype(system)})
end

@inline set_particle_density!(v, system, particle, density) = v

@inline set_particle_pressure!(v, system, particle, pressure) = v

@inline function smoothing_kernel(system, distance)
    (; smoothing_kernel, smoothing_length) = system
    return kernel(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_deriv(system, distance)
    (; smoothing_kernel, smoothing_length) = system
    return kernel_deriv(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_grad(system, pos_diff, distance)
    return kernel_grad(system.smoothing_kernel, pos_diff, distance, system.smoothing_length)
end

@inline function smoothing_kernel_grad(system::BoundarySystem, pos_diff, distance)
    (; smoothing_kernel, smoothing_length) = system.boundary_model

    return kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
end

@inline function smoothing_kernel_grad(system, pos_diff, distance, particle)
    return corrected_kernel_grad(system.smoothing_kernel, pos_diff, distance,
                                 system.smoothing_length, system.correction, system,
                                 particle)
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

function update_final!(system, v, u, v_ode, u_ode, semi, t; update_from_callback=false)
    return system
end

# Only for systems requiring a mandatory callback
reset_callback_flag!(system) = system
