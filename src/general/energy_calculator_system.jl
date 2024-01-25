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
                           neighbor_system, semi; timer_str="")
    @trixi_timeit timer() "calculate energy" begin
        dv = wrap_v(dv_ode, system, semi)
        dv_neighbor = wrap_v(dv_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        for particle in eachparticle(neighbor_system)
            d_velocity = current_velocity(dv_neighbor, neighbor_system, particle)
            velocity = current_velocity(v_neighbor, neighbor_system, particle)

            dv[1] -= dot(hydrodynamic_mass(neighbor_system, particle) * d_velocity,
                         velocity)
        end
    end

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
