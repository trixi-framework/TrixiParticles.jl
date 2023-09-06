"""
    trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="", custom_quantities...)

Converts Trixi simulation data to VTK format.

# Arguments
- `vu_ode::ODESolution`: Solution of the Trixi ODE system.
- `semi::SemiDiscretization`: Semi-discretization of the TrixiParticles simulation.
- `t::Float64`: Current time of the simulation.
- `iter::Union{Nothing, Int}` (optional): Iteration number. Defaults to `nothing`.
- `output_directory::AbstractString` (optional): Output directory path. Defaults to `"out"`.
- `prefix::AbstractString` (optional): Prefix for output files. Defaults to an empty string.
- `custom_quantities...`: Additional custom quantities to include in the VTK output.

# Details
This function converts TrixiParticles simulation data to VTK format.
It iterates over each system in the semi-discretization and calls the overloaded `trixi2vtk` function for each system.

# Example
```julia
trixi2vtk(vu, semi, 0.0, iter=1, output_directory="output", prefix="solution", velocity=compute_velocity)
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

"""
    trixi2vtk(v, u, t, system; output_directory="out", prefix="", iter=nothing,
    system_name=vtkname(system), custom_quantities...)

Converts a single Trixi system data to VTK format.

# Arguments
- `v::AbstractVector`: Vector of solution variables.
- `u::AbstractVector`: Vector of unknowns.
- `t::Float64`: Current time of the simulation.
- `system::TrixiSystem`: Trixi system to convert to VTK.
- `output_directory::AbstractString` (optional): Output directory path. Defaults to "out".
- `prefix::AbstractString` (optional): Prefix for output files. Defaults to an empty string.
- `iter::Union{Nothing, Int}` (optional): Iteration number. Defaults to nothing.
- `system_name::AbstractString` (optional): Name of the system for the output file. Defaults to the VTK name of the system.
- `custom_quantities...`: Additional custom quantities to include in the VTK output.

# Details
This function converts a single Trixi system's data to VTK format. It creates the necessary VTK file,
writes the solution variables and unknowns to the file, and includes additional custom quantities if provided.

# Example
trixi2vtk(v, u, 0.0, fluid_system, output_directory="output", prefix="solution", velocity=compute_velocity)

"""
function trixi2vtk(v, u, t, system, periodic_box; output_directory="out", prefix="",
                   iter=nothing, system_name=vtkname(system), custom_value=nothing,
                   custom_quantities...)
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

        if custom_value !== nothing
            for (key, value) in custom_value
                if axes(value, 1) == ndims(system)
                    vtk[string(key)] = view(value, 1:ndims(system), :)
                else
                    vtk[string(key)] = value
                end
            end
        end

        # Extract custom quantities for this system
        for (key, func) in custom_quantities
            value = func(v, u, t, system)
            if value !== nothing
                vtk[string(key)] = func(v, u, t, system)
            end
        end
    end
end

"""
    trixi2vtk(coordinates; output_directory="out", prefix="", filename="coordinates")

Converts coordinate data to VTK format.

# Arguments
- `coordinates::AbstractMatrix`: Matrix of coordinate data.
- `output_directory::AbstractString` (optional): Output directory path. Defaults to `"out"`.
- `prefix::AbstractString` (optional): Prefix for the output file. Defaults to an empty string.
- `filename::AbstractString` (optional): Name of the output file. Defaults to `"coordinates"`.

# Details
This function converts coordinate data to VTK format.
It creates a VTK file with the specified filename and saves the coordinate data as points in the VTK file.
Each coordinate is treated as a vertex cell in the VTK file.

# Returns
- `file::AbstractString`: Path to the generated VTK file.

# Example
```julia
coordinates = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
vtk_file = trixi2vtk(coordinates, output_directory="output", prefix="data", filename="coords")
"""
function trixi2vtk(coordinates; output_directory="out", prefix="", filename="coordinates")
    mkpath(output_directory)
    file = prefix === "" ? joinpath(output_directory, filename) :
           joinpath(output_directory, "$(prefix)_$filename")

    points = coordinates
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]
    vtk_grid(vtk -> nothing, file, points, cells)

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
