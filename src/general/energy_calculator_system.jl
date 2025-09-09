struct EnergyCalculatorSystem{NDIMS, ELTYPE <: Real} <: System{NDIMS}
    initial_energy::ELTYPE
    current_energy::Vector{ELTYPE}

    function EnergyCalculatorSystem{NDIMS}(; initial_energy=0.0) where {NDIMS}
        new{NDIMS, typeof(initial_energy)}(initial_energy, [initial_energy])
    end
end

timer_name(::EnergyCalculatorSystem) = "dummy"

@inline Base.eltype(::EnergyCalculatorSystem{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

@inline nparticles(::EnergyCalculatorSystem) = 1

@inline v_nvariables(::EnergyCalculatorSystem) = 1

@inline initial_coordinates(::EnergyCalculatorSystem) = nothing

@inline particle_spacing(::EnergyCalculatorSystem, particle) = 0.0

function write_u0!(u0, ::EnergyCalculatorSystem)
    # `u` can't be of size 0, or we have to write an extra `write2vtk` function
    u0 .= 0

    return u0
end

function write_v0!(v0, system::EnergyCalculatorSystem)
    v0[1] = system.initial_energy

    return v0
end

function nhs_coords(system, neighbor::EnergyCalculatorSystem, u)
    return u
end

function update_quantities!(system::EnergyCalculatorSystem, v, u, v_ode, u_ode, semi, t)
    system.current_energy[1] = v[1]

    return system
end

@inline function interact!(dv_ode, v_ode, u_ode, system::EnergyCalculatorSystem,
                           neighbor_system::SolidSystem, semi; timer_str="")
    @trixi_timeit timer() "calculate energy" begin
        dv = wrap_v(dv_ode, system, semi)
        dv_fixed = neighbor_system.cache.dv_fixed

        n_fixed_particles = nparticles(neighbor_system) - n_moving_particles(neighbor_system)
        for fixed_particle in 1:n_fixed_particles
            particle = fixed_particle + n_moving_particles(neighbor_system)
            velocity = current_velocity(nothing, neighbor_system, particle)
            dv_particle = current_velocity(dv_fixed, neighbor_system, fixed_particle)

            # The force on the fixed particle is mass times acceleration
            F_particle = neighbor_system.mass[particle] * dv_particle

            # To obtain energy, we need to integrate the instantaneous power.
            # Instantaneous power is force done BY the particle times prescribed velocity.
            # The work done BY the particle is the negative of the work done ON it.
            dv[1] -= dot(F_particle, velocity)
        end
    end

    return dv_ode
end

@inline function interact!(dv_ode, v_ode, u_ode, system::EnergyCalculatorSystem,
                           neighbor_system, semi; timer_str="")
    return dv_ode
end

@inline function interact!(dv_ode, v_ode, u_ode, system::EnergyCalculatorSystem,
                           neighbor_system::EnergyCalculatorSystem, semi; timer_str="")
    return dv_ode
end

@inline function interact!(dv_ode, v_ode, u_ode, system,
                           neighbor_system::EnergyCalculatorSystem, semi; timer_str="")
    return dv_ode
end

vtkname(system::EnergyCalculatorSystem) = "energy"

function write2vtk!(vtk, v, u, t, system::EnergyCalculatorSystem;
                    write_meta_data=true)
    vtk["energy", VTKFieldData()] = v[1]
    vtk["initial_energy"] = system.initial_energy

    return vtk
end

function system_data(system::EnergyCalculatorSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    return (; energy=v[1], instantaneous_power=dv[1])
end

function Base.show(io::IO, system::EnergyCalculatorSystem)
    print(io, "EnergyCalculatorSystem")
end

function Base.show(io::IO, ::MIME"text/plain", system::EnergyCalculatorSystem)
    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "EnergyCalculatorSystem")
        summary_footer(io)
    end
end
