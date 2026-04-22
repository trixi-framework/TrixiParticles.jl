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

function update_positions!(system::AbstractFluidSystem, v, u, v_ode, u_ode, semi, t)
    nhs = get_neighborhood_search(system, semi)

    # `GridNeighborhoodSearch` with a `FullGridCellList` requires a bounding box.
    # This function deactivates particles that move outside the bounding box to prevent
    # simulation crashes.
    # Note that deactivating particles is only possible in combination with a 'SystemBuffer'.
    deactivate_out_of_bounds_particles!(system, buffer(system), nhs, v, u, semi)
end

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
@inline function continuity_equation!(drho_particle, ::SummationDensity,
                                      particle_system, neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
    return drho_particle
end

# This formulation was chosen to be consistent with the used pressure_acceleration formulations
@propagate_inbounds function continuity_equation!(drho_particle,
                                                  ::ContinuityDensity,
                                                  particle_system::AbstractFluidSystem,
                                                  neighbor_system,
                                                  particle, neighbor, pos_diff, distance,
                                                  m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
    continuity_equation!(drho_particle, particle_system, neighbor_system,
                         particle, neighbor, pos_diff, distance,
                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
end

@propagate_inbounds function continuity_equation!(drho_particle,
                                                  particle_system, neighbor_system,
                                                  particle, neighbor, pos_diff, distance,
                                                  m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
    v_diff = v_a - v_b

    v_diff = continuity_equation_shifting_term(v_diff,
                                               shifting_technique(particle_system),
                                               particle_system, neighbor_system,
                                               particle, neighbor, rho_a, rho_b)

    # Since this is one of the most performance critical functions, using fast divisions
    # here gives a significant speedup on GPUs.
    # See the docs page "Development" for more details on `div_fast`.
    drho_particle[] += div_fast(rho_a, rho_b) * m_b * dot(v_diff, grad_kernel)

    # Artificial density diffusion should only be applied to systems representing a fluid
    # with the same physical properties i.e. density and viscosity.
    # TODO: shouldn't be applied to particles on the interface (depends on PR #539)
    if particle_system === neighbor_system
        density_diffusion!(drho_particle, density_diffusion(particle_system),
                           particle_system, particle, neighbor,
                           pos_diff, distance, m_b, rho_a, rho_b, grad_kernel)
    end

    return drho_particle
end

@inline function write_drho_particle!(dv, density_calculator, drho_particle, particle)
    return dv
end

@propagate_inbounds function write_drho_particle!(dv, ::ContinuityDensity,
                                                  drho_particle, particle)
    dv[end, particle] += drho_particle[]

    return dv
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

function restart_u(system::AbstractFluidSystem, data)
    coords_total = zeros(coordinates_eltype(system), u_nvariables(system),
                         n_integrated_particles(system))
    coords_total .= coordinates_eltype(system)(1e16)

    coords_active = data.coordinates

    for particle in axes(coords_active, 2)
        for dim in 1:ndims(system)
            coords_total[dim, particle] = coords_active[dim, particle]
        end
    end

    if !isnothing(buffer(system))
        system.buffer.active_particle .= false
        system.buffer.active_particle[1:size(coords_active, 2)] .= true
    end

    update_system_buffer!(system.buffer)

    return coords_total
end

function restart_v(system::AbstractFluidSystem, data)
    velocity_total = zeros(eltype(system), v_nvariables(system),
                           n_integrated_particles(system))
    velocity_active = zeros(eltype(system), v_nvariables(system), size(data.velocity, 2))

    velocity_active[1:ndims(system), :] = data.velocity
    write_density_and_pressure!(velocity_active, system, density_calculator(system),
                                data.pressure, data.density)

    for particle in axes(velocity_active, 2)
        for i in axes(velocity_active, 1)
            velocity_total[i, particle] = velocity_active[i, particle]
        end
    end

    return velocity_total
end

function check_configuration(fluid_system::AbstractFluidSystem, systems, nhs)
    if !(fluid_system isa ParticlePackingSystem) && !isnothing(fluid_system.surface_tension)
        foreach_system(systems) do neighbor
            if neighbor isa AbstractFluidSystem &&
               isnothing(fluid_system.surface_tension) &&
               isnothing(fluid_system.surface_normal_method)
                throw(ArgumentError("either none or all fluid systems in a simulation need " *
                                    "to use a surface tension model or a surface normal method."))
            end
        end
    end
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
