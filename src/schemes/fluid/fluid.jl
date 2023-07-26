abstract type FluidSystem{NDIMS} <: System{NDIMS} end

timer_name(::FluidSystem) = "fluid"

@inline hydrodynamic_mass(system::FluidSystem, particle) = system.mass[particle]

@inline function particle_pressure(v, system, particle)
    return system.pressure[particle]
end

function write_u0!(u0, system::FluidSystem)
    @unpack initial_condition = system

    for particle in eachparticle(system)
        # Write particle coordinates
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_condition.coordinates[dim, particle]
        end
    end

    return u0
end

include("viscosity.jl")
include("transport_velocity.jl")
include("weakly_compressible_sph/weakly_compressible_sph.jl")
include("entropically_damped_sph/entropically_damped_sph.jl")
