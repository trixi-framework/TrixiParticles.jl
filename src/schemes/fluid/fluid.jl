@inline hydrodynamic_mass(system::FluidSystem, particle) = system.mass[particle]

function write_u0!(u0, system::FluidSystem)
    (; initial_condition) = system

    for particle in eachparticle(system)
        # Write particle coordinates
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_condition.coordinates[dim, particle]
        end
    end

    return u0
end

@inline viscosity_model(system::FluidSystem) = system.viscosity

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
include("weakly_compressible_sph/weakly_compressible_sph.jl")
include("entropically_damped_sph/entropically_damped_sph.jl")
