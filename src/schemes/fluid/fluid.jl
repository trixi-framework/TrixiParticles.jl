@inline function each_moving_particle(system::FluidSystem)
    return each_moving_particle(system, system.buffer)
end

@inline function active_coordinates(u, system::FluidSystem)
    return active_coordinates(u, system, system.buffer)
end

@inline active_particles(system::FluidSystem)  = active_particles(system, system.buffer)

update_system_buffer!(system::FluidSystem) = update!(system.buffer)

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

@inline viscosity_model(system, neighbor_system::FluidSystem) = neighbor_system.viscosity

include("pressure_acceleration.jl")
include("viscosity.jl")
include("weakly_compressible_sph/weakly_compressible_sph.jl")
include("entropically_damped_sph/entropically_damped_sph.jl")
