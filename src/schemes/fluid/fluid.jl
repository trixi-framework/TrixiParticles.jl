# WARNING!
# These functions are intended to be used internally to set the density
# of newly activated particles in a callback.
# DO NOT use outside a callback. OrdinaryDiffEq does not allow changing `v` and `u`
# outside of callbacks.
@inline function set_particle_density!(v, system::AbstractFluidSystem, particle, density)
    current_density(v, system)[particle] = density

    return v
end

# WARNING!
# These functions are intended to be used internally to set the pressure
# of newly activated particles in a callback.
# DO NOT use outside a callback. OrdinaryDiffEq does not allow changing `v` and `u`
# outside of callbacks.
@inline function set_particle_pressure!(v, system::AbstractFluidSystem, particle, pressure)
    current_pressure(v, system)[particle] = pressure

    return v
end

function create_cache_density(initial_condition, ::SummationDensity)
    density = similar(initial_condition.density)

    return (; density)
end

function create_cache_density(ic, ::ContinuityDensity)
    # Density in this case is added to the end of `v` and allocated by modifying `v_nvariables`.
    return (;)
end

function create_cache_refinement(initial_condition, ::Nothing, smoothing_length)
    smoothing_length_factor = smoothing_length / initial_condition.particle_spacing
    return (; smoothing_length, smoothing_length_factor)
end

# TODO
function create_cache_refinement(initial_condition, refinement, smoothing_length)
    # TODO: If refinement is not `Nothing` and `correction` is not `Nothing`, then throw an error
end

@propagate_inbounds function hydrodynamic_mass(system::AbstractFluidSystem, particle)
    return system.mass[particle]
end

function smoothing_length(system::AbstractFluidSystem, particle)
    return smoothing_length(system, system.particle_refinement, particle)
end

function smoothing_length(system::AbstractFluidSystem, ::Nothing, particle)
    return system.cache.smoothing_length
end

function initial_smoothing_length(system::AbstractFluidSystem)
    return initial_smoothing_length(system, system.particle_refinement)
end

initial_smoothing_length(system, ::Nothing) = system.cache.smoothing_length

function initial_smoothing_length(system, refinement)
    # TODO
    return system.cache.initial_smoothing_length_factor *
           system.initial_condition.particle_spacing
end

@inline function particle_spacing(system::AbstractFluidSystem, particle)
    return particle_spacing(system, system.particle_refinement, particle)
end

@inline particle_spacing(system, ::Nothing, _) = system.initial_condition.particle_spacing

@inline function particle_spacing(system, refinement, particle)
    (; smoothing_length_factor) = system.cache
    return smoothing_length(system, particle) / smoothing_length_factor
end

function write_u0!(u0, system::AbstractFluidSystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_condition.coordinates)
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

function write_v0!(v0, system::AbstractFluidSystem)
    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(system.initial_condition.velocity)
    copyto!(v0, indices, system.initial_condition.velocity, indices)

    write_v0!(v0, system, system.density_calculator)

    return v0
end

write_v0!(v0, system::AbstractFluidSystem, _) = v0

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.

@inline function viscosity_model(system::AbstractFluidSystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::AbstractFluidSystem,
                                 neighbor_system::AbstractBoundarySystem)
    return neighbor_system.boundary_model.viscosity
end

@inline system_state_equation(system::AbstractFluidSystem) = system.state_equation

@inline acceleration_source(system::AbstractFluidSystem) = system.acceleration

function compute_density!(system, u, u_ode, semi, ::ContinuityDensity)
    # No density update with `ContinuityDensity`
    return system
end

function compute_density!(system, u, u_ode, semi, ::SummationDensity)
    (; cache) = system
    (; density) = cache # Density is in the cache for SummationDensity

    summation_density!(system, semi, u, u_ode, density)
end

# With 'SummationDensity', density is calculated in wcsph/system.jl:compute_density!
@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      particle_system, neighbor_system,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b, grad_kernel)
    return dv
end

# This formulation was chosen to be consistent with the used pressure_acceleration formulations
@propagate_inbounds function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                                  particle_system::AbstractFluidSystem,
                                                  neighbor_system,
                                                  v_particle_system, v_neighbor_system,
                                                  particle, neighbor, pos_diff, distance,
                                                  m_b, rho_a, rho_b, grad_kernel)
    vdiff = current_velocity(v_particle_system, particle_system, particle) -
            current_velocity(v_neighbor_system, neighbor_system, neighbor)

    vdiff += continuity_equation_shifting_term(shifting_technique(particle_system),
                                               particle_system, neighbor_system,
                                               particle, neighbor, rho_a, rho_b)

    dv[end, particle] += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)

    # Artificial density diffusion should only be applied to systems representing a fluid
    # with the same physical properties i.e. density and viscosity.
    # TODO: shouldn't be applied to particles on the interface (depends on PR #539)
    if particle_system === neighbor_system
        density_diffusion!(dv, density_diffusion(particle_system),
                           v_particle_system, particle, neighbor,
                           pos_diff, distance, m_b, rho_a, rho_b, particle_system,
                           grad_kernel)
    end
end

function calculate_dt(v_ode, u_ode, cfl_number, system::AbstractFluidSystem, semi)
    (; viscosity, acceleration, surface_tension) = system

    # TODO variable smoothing length
    smoothing_length_ = initial_smoothing_length(system)

    dt_viscosity = 0.125 * smoothing_length_^2
    if !isnothing(system.viscosity)
        dt_viscosity = dt_viscosity /
                       kinematic_viscosity(system, viscosity, smoothing_length_,
                                           system_sound_speed(system))
    end

    # TODO Adami et al. (2012) just use the gravity here, but Antuono et al. (2012)
    # are using a per-particle acceleration. Is that supposed to be the previous RHS?
    # Morris (2000) also uses the acceleration and cites Monaghan (1992)
    dt_acceleration = 0.25 * sqrt(smoothing_length_ / norm(acceleration))

    # TODO Everyone seems to be doing this differently.
    # Morris (2000) uses the additional condition CFL < 0.25.
    # Sun et al. (2017) only use h / c (because c depends on v_max as c >= 10 v_max).
    # Adami et al. (2012) use h / (c + v_max) with a fixed CFL of 0.25.
    # Antuono et al. (2012) use h / (c + v_max + h * pi_max), where pi is the viscosity coefficient.
    # Antuono et al. (2015) use h / (c + h * pi_max).
    #
    # See docstring of the callback for the references.
    dt_sound_speed = cfl_number * smoothing_length_ / system_sound_speed(system)

    # Eq. 28 in Morris (2000)
    dt = min(dt_viscosity, dt_acceleration, dt_sound_speed)
    if surface_tension isa SurfaceTensionMorris ||
       surface_tension isa SurfaceTensionMomentumMorris
        v = wrap_v(v_ode, system, semi)
        dt_surface_tension = sqrt(current_density(v, system, 1) * smoothing_length_^3 /
                                  (2 * pi * surface_tension.surface_tension_coefficient))
        dt = min(dt, dt_surface_tension)
    end

    return dt
end

@inline function surface_tension_model(system::AbstractFluidSystem)
    return system.surface_tension
end

@inline function surface_tension_model(system)
    return nothing
end

@inline function surface_normal_method(system::AbstractFluidSystem)
    return system.surface_normal_method
end

@inline function surface_normal_method(system)
    return nothing
end

function system_data(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    (; mass) = system

    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    acceleration = current_velocity(dv, system)
    density = current_density(v, system)
    pressure = current_pressure(v, system)

    return (; coordinates, velocity, mass, density, pressure, acceleration)
end

function available_data(::AbstractFluidSystem)
    return (:coordinates, :velocity, :mass, :density, :pressure, :acceleration)
end

include("pressure_acceleration.jl")
include("viscosity.jl")
include("shifting_techniques.jl")
include("surface_tension.jl")
include("surface_normal_sph.jl")
include("weakly_compressible_sph/weakly_compressible_sph.jl")
include("entropically_damped_sph/entropically_damped_sph.jl")
