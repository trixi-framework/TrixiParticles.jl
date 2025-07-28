@inline function set_particle_density!(v, system::FluidSystem, particle, density)
    set_particle_density!(v, system, system.density_calculator, particle, density)
end

function create_cache_resize(n_particles)
    additional_capacity = Ref(0)
    delete_candidates = Bool[]
    values_conserved = Ref(false)

    return (; additional_capacity, delete_candidates, values_conserved)
end

# WARNING!
# These functions are intended to be used internally to set the density
# of newly activated particles in a callback.
# DO NOT use outside a callback. OrdinaryDiffEq does not allow changing `v` and `u`
# outside of callbacks.
@inline set_particle_density!(v, system, ::SummationDensity, particle, density) = v

@inline function set_particle_density!(v, system, ::ContinuityDensity, particle, density)
    v[end, particle] = density

    return v
end

function create_cache_density(initial_condition, ::SummationDensity)
    density = copy(initial_condition.density)

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

function create_cache_refinement(initial_condition, refinement, smoothing_length)
    smoothng_length_factor = smoothing_length / initial_condition.particle_spacing

    beta = Vector{eltype(initial_condition)}(undef, nparticles(initial_condition))

    return (; smoothing_length=smoothing_length * ones(length(initial_condition.density)),
            smoothing_length_factor=smoothng_length_factor, beta=beta)
end

@propagate_inbounds hydrodynamic_mass(system::FluidSystem, particle) = system.mass[particle]

function smoothing_length(system::FluidSystem, particle)
    return smoothing_length(system, system.particle_refinement, particle)
end

function smoothing_length(system::FluidSystem, ::Nothing, particle)
    return system.cache.smoothing_length
end

function smoothing_length(system::FluidSystem, refinement, particle)
    return system.cache.smoothing_length[particle]
end

function initial_smoothing_length(system::FluidSystem)
    return initial_smoothing_length(system, system.particle_refinement)
end

initial_smoothing_length(system, ::Nothing) = system.cache.smoothing_length

function initial_smoothing_length(system, refinement)
    # TODO
    return system.cache.initial_smoothing_length_factor *
           system.initial_condition.particle_spacing
end

@inline function particle_spacing(system::FluidSystem, particle)
    return particle_spacing(system, system.particle_refinement, particle)
end

@inline particle_spacing(system, ::Nothing, _) = system.initial_condition.particle_spacing

@inline function particle_spacing(system, refinement, particle)
    (; smoothing_length_factor) = system.cache
    return smoothing_length(system, particle) / smoothing_length_factor
end

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

@inline function viscosity_model(system::FluidSystem, neighbor_system::FluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::FluidSystem, neighbor_system::BoundarySystem)
    return neighbor_system.boundary_model.viscosity
end

@inline system_state_equation(system::FluidSystem) = system.state_equation

function compute_density!(system, u, u_ode, semi, ::ContinuityDensity)
    # No density update with `ContinuityDensity`
    return system
end

function compute_density!(system, u, u_ode, semi, ::SummationDensity)
    (; cache) = system
    (; density) = cache # Density is in the cache for SummationDensity

    summation_density!(system, semi, u, u_ode, density)
end

function calculate_dt(v_ode, u_ode, cfl_number, system::FluidSystem, semi)
    (; viscosity, acceleration, surface_tension) = system

    # TODO
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

@inline function surface_tension_model(system::FluidSystem)
    return system.surface_tension
end

@inline function surface_tension_model(system)
    return nothing
end

@inline function surface_normal_method(system::FluidSystem)
    return system.surface_normal_method
end

@inline function surface_normal_method(system)
    return nothing
end

function system_data(system::FluidSystem, v_ode, u_ode, semi)
    (; mass) = system

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    density = current_density(v, system)
    pressure = current_pressure(v, system)

    return (; coordinates, velocity, mass, density, pressure)
end

function available_data(::FluidSystem)
    return (:coordinates, :velocity, :mass, :density, :pressure)
end

include("pressure_acceleration.jl")
include("viscosity.jl")
include("transport_velocity.jl")
include("surface_tension.jl")
include("surface_normal_sph.jl")
include("weakly_compressible_sph/weakly_compressible_sph.jl")
include("entropically_damped_sph/entropically_damped_sph.jl")

@inline function add_velocity!(du, v, particle,
                               system::Union{EntropicallyDampedSPHSystem,
                                             WeaklyCompressibleSPHSystem})
    add_velocity!(du, v, particle, system, system.transport_velocity)
end

@inline function momentum_convection(system, neighbor_system, pos_diff, distance,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    return zero(grad_kernel)
end

@inline function momentum_convection(system,
                                     neighbor_system::Union{EntropicallyDampedSPHSystem,
                                                            WeaklyCompressibleSPHSystem},
                                     pos_diff, distance,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    momentum_convection(system, neighbor_system, system.transport_velocity,
                        system.particle_refinement, pos_diff, distance, v_particle_system,
                        v_neighbor_system, rho_a, rho_b, m_a, m_b, particle, neighbor,
                        grad_kernel)
end

function update_final!(system::FluidSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    # Check if `UpdateCallback` is used when simulating with TVF
    update_final!(system, system.transport_velocity,
                  v, u, v_ode, u_ode, semi, t; update_from_callback)

    # Compute correction factor when using particle refinement
    compute_beta_correction!(system, system.particle_refinement, v_ode, u_ode, semi)

    return system
end

compute_beta_correction!(system, ::Nothing, v_ode, u_ode, semi) = system

function compute_beta_correction!(system, refinement, v_ode, u_ode, semi)
    (; beta) = system.cache

    set_zero!(beta)

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)

        # Loop over all pairs of particles and neighbors within the kernel cutoff
        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               neighborhood_search) do particle, neighbor,
                                                       pos_diff, distance
            rho_a = particle_density(v, system, particle)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)

            W_deriv = kernel_deriv(system.smoothing_kernel, distance,
                                   smoothing_length(system, particle))

            beta[particle] -= m_b * distance * W_deriv * (rho_a * ndims(system))
        end
    end

    return system
end
