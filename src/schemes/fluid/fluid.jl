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

function write_v0!(v0, system::FluidSystem)
    write_v0!(v0, system, system.transport_velocity)
end

function write_v0!(v0, system::FluidSystem, ::Nothing)
    for particle in eachparticle(system)
        # Write particle velocities
        v_init = initial_velocity(system, particle)
        for dim in 1:ndims(system)
            v0[dim, particle] = v_init[dim]
        end
    end

    write_v0!(v0, system.density_calculator, system)

    return v0
end

@inline function initial_velocity(system, particle)
    initial_velocity(system, particle, system.initial_velocity_function)
end

@inline function initial_velocity(system, particle, ::Nothing)
    return extract_svector(system.initial_condition.velocity, system, particle)
end

@inline function initial_velocity(system, particle, init_velocity_function)
    position = initial_coords(system, particle)

    return SVector(ntuple(i -> init_velocity_function[i](position, 0), Val(ndims(system))))
end

@inline viscosity_model(system::FluidSystem) = system.viscosity

include("pressure_acceleration.jl")
include("viscosity.jl")
include("transport_velocity.jl")
include("weakly_compressible_sph/weakly_compressible_sph.jl")
include("entropically_damped_sph/entropically_damped_sph.jl")
