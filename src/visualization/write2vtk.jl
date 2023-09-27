"""
    trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="", custom_quantities...)

Convert Trixi simulation data to VTK format.

# Arguments
- `vu_ode`: Solution of the TrixiParticles ODE system at one time step. This expects an `ArrayPartition` as returned in the examples as `sol`.
- `semi`:   Semidiscretization of the TrixiParticles simulation.
- `t`:      Current time of the simulation.

# Keywords
- `iter`:                 Iteration number when multiple iterations are to be stored in separate files.
- `output_directory`:     Output directory path. Defaults to `"out"`.
- `prefix`:               Prefix for output files. Defaults to an empty string.
- `custom_quantities...`: Additional custom quantities to include in the VTK output. TODO.


# Example
```julia
trixi2vtk(sol[end], semi, 0.0, iter=1, output_directory="output", prefix="solution")

TODO: example for custom_quantities
"""
function trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="",
                   custom_quantities...)
    (; systems, neighborhood_searches) = semi
    v_ode, u_ode = vu_ode.x

    # Add `_i` to each system name, where `i` is the index of the corresponding
    # system type.
    # `["fluid", "boundary", "boundary"]` becomes `["fluid_1", "boundary_1", "boundary_2"]`.
    cnames = systems .|> vtkname
    filenames = [string(cnames[i], "_", count(==(cnames[i]), cnames[1:i]))
                 for i in eachindex(cnames)]

    foreach_enumerate(systems) do (system_index, system)
        v = wrap_v(v_ode, system_index, system, semi)
        u = wrap_u(u_ode, system_index, system, semi)
        periodic_box = neighborhood_searches[system_index][system_index].periodic_box

        trixi2vtk(v, u, t, system, periodic_box;
                  output_directory=output_directory,
                  system_name=filenames[system_index], iter=iter, prefix=prefix,
                  custom_quantities...)
    end
end

# Convert data for a single TrixiParticle system to VTK format
function trixi2vtk(v, u, t, system, periodic_box; output_directory="out", prefix="",
                   iter=nothing, system_name=vtkname(system), custom_quantities...)
    mkpath(output_directory)

    # handle "_" on optional pre/postfix strings
    add_opt_str_pre(str) = (str === "" ? "" : "$(str)_")
    add_opt_str_post(str) = (str === nothing ? "" : "_$(str)")

    file = joinpath(output_directory,
                    add_opt_str_pre(prefix) * "$system_name"
                    * add_opt_str_post(iter))

    points = periodic_coords(current_coordinates(u, system), periodic_box)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(file, points, cells) do vtk
        write2vtk!(vtk, v, u, t, system)

        # Store particle index
        vtk["index"] = eachparticle(system)

        # Extract custom quantities for this system
        for (key, quantity) in custom_quantities
            value = custom_quantity(quantity, v, u, t, system)
            if value !== nothing
                vtk[string(key)] = value
            end
        end
    end
end

function custom_quantity(quantity::AbstractArray, v, u, t, system)
    return quantity
end

function custom_quantity(quantity, v, u, t, system)
    # Assume `quantity` is a function of `v`, `u`, `t`, and `system`
    return quantity(v, u, t, system)
end

"""
    trixi2vtk(coordinates; output_directory="out", prefix="", filename="coordinates")

Convert coordinate data to VTK format.

# Arguments
- `coordinates`:                 Coordinates to be saved.
- `output_directory` (optional): Output directory path. Defaults to `"out"`.
- `prefix` (optional):           Prefix for the output file. Defaults to an empty string.
- `filename` (optional):         Name of the output file. Defaults to `"coordinates"`.

# Returns
- `file::AbstractString`: Path to the generated VTK file.
"""
function trixi2vtk(coordinates; output_directory="out", prefix="", filename="coordinates",
                   custom_quantities...)
    mkpath(output_directory)
    file = prefix === "" ? joinpath(output_directory, filename) :
           joinpath(output_directory, "$(prefix)_$filename")

    points = coordinates
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(file, points, cells) do vtk
        # Store particle index.
        vtk["index"] = [i for i in axes(coordinates, 2)]

        # Extract custom quantities for this system.
        for (key, quantity) in custom_quantities
            if quantity !== nothing
                vtk[string(key)] = quantity
            end
        end
    end

    return file
end

vtkname(system::FluidSystem) = "fluid"
vtkname(system::TotalLagrangianSPHSystem) = "solid"
vtkname(system::BoundarySPHSystem) = "boundary"

function write2vtk!(vtk, v, u, t, system::FluidSystem)
    vtk["velocity"] = view(v, 1:ndims(system), :)
    vtk["density"] = [particle_density(v, system, particle)
                      for particle in eachparticle(system)]
    vtk["pressure"] = [particle_pressure(v, system, particle)
                       for particle in eachparticle(system)]

    return vtk
end

function write2vtk!(vtk, v, u, t, system::TotalLagrangianSPHSystem)
    n_fixed_particles = nparticles(system) - n_moving_particles(system)

    vtk["velocity"] = hcat(view(v, 1:ndims(system), :),
                           zeros(ndims(system), n_fixed_particles))
    vtk["material_density"] = system.material_density

    write2vtk!(vtk, v, u, t, system.boundary_model, system)
end

function write2vtk!(vtk, v, u, t, system::BoundarySPHSystem)
    write2vtk!(vtk, v, u, t, system.boundary_model, system)
end

function write2vtk!(vtk, v, u, t, model, system)
    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles, system)
    write2vtk!(vtk, v, u, t, model, model.viscosity, system)
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles, viscosity, system)
    vtk["hydrodynamic_density"] = [particle_density(v, system, particle)
                                   for particle in eachparticle(system)]
    vtk["pressure"] = model.pressure

    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles,
                    viscosity::ViscosityAdami, system)
    vtk["hydrodynamic_density"] = [particle_density(v, system, particle)
                                   for particle in eachparticle(system)]
    vtk["pressure"] = model.pressure
    vtk["wall_velocity"] = view(model.cache.wall_velocity, 1:ndims(system), :)

    return vtk
end
