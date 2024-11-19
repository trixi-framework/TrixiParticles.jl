@inline function set_particle_density!(v, system::FluidSystem, particle, density)
    set_particle_density!(v, system, system.density_calculator, particle, density)
end

function create_cache_density(initial_condition, ::SummationDensity)
    density = similar(initial_condition.density)

    return (; density)
end

function create_cache_density(ic, ::ContinuityDensity)
    # Density in this case is added to the end of `v` and allocated by modifying `v_nvariables`.
    return (;)
end

@propagate_inbounds hydrodynamic_mass(system::FluidSystem, particle) = system.mass[particle]

function write_u0!(u0, system::FluidSystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_condition.coordinates)
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

function write_v0!(v0, system::FluidSystem)
    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(system.initial_condition.velocity)
    copyto!(v0, indices, system.initial_condition.velocity, indices)

    write_v0!(v0, system, system.density_calculator)
    write_v0!(v0, system, system.transport_velocity)

    return v0
end

write_v0!(v0, system::FluidSystem, _) = v0

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline viscosity_model(system::FluidSystem, neighbor_system::FluidSystem) = neighbor_system.viscosity
@inline viscosity_model(system::FluidSystem, neighbor_system::BoundarySystem) = neighbor_system.boundary_model.viscosity

function compute_density!(system, u, u_ode, semi, ::ContinuityDensity)
    # No density update with `ContinuityDensity`
    return system
end

function compute_density!(system, u, u_ode, semi, ::SummationDensity)
    (; cache) = system
    (; density) = cache # Density is in the cache for SummationDensity

    summation_density!(system, semi, u, u_ode, density)
end

function calculate_dt(v_ode, u_ode, cfl_number, system::FluidSystem)
    (; smoothing_length, viscosity, acceleration) = system

    dt_viscosity = 0.125 * smoothing_length^2 / kinematic_viscosity(system, viscosity)

    # TODO Adami et al. (2012) just use the gravity here, but Antuono et al. (2012)
    # are using a per-particle acceleration. Is that supposed to be the previous RHS?
    dt_acceleration = 0.25 * sqrt(smoothing_length / norm(acceleration))

    # TODO Everyone seems to be doing this differently.
    # Sun et al. (2017) only use h / c (because c depends on v_max as c >= 10 v_max).
    # Adami et al. (2012) use h / (c + v_max) with a fixed CFL of 0.25.
    # Antuono et al. (2012) use h / (c + v_max + h * pi_max), where pi is the viscosity coefficient.
    # Antuono et al. (2015) use h / (c + h * pi_max).
    #
    # See docstring of the callback for the references.
    dt_sound_speed = cfl_number * smoothing_length / system_sound_speed(system)

    return min(dt_viscosity, dt_acceleration, dt_sound_speed)
end

include("pressure_acceleration.jl")
include("viscosity.jl")
include("transport_velocity.jl")
include("surface_tension.jl")
include("weakly_compressible_sph/weakly_compressible_sph.jl")
include("entropically_damped_sph/entropically_damped_sph.jl")

@inline function add_velocity!(du, v, particle,
                               system::Union{EntropicallyDampedSPHSystem,
                                             WeaklyCompressibleSPHSystem})
    add_velocity!(du, v, particle, system, system.transport_velocity)
end

@inline function momentum_convection(system, neighbor_system,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    return zero(grad_kernel)
end

@inline function momentum_convection(system,
                                     neighbor_system::Union{EntropicallyDampedSPHSystem,
                                                            WeaklyCompressibleSPHSystem},
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    momentum_convection(system, neighbor_system, system.transport_velocity,
                        v_particle_system, v_neighbor_system, rho_a, rho_b,
                        m_a, m_b, particle, neighbor, grad_kernel)
end
