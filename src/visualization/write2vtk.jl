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

function trixi2vtk(v, u, t, system, periodic_box; output_directory="out", prefix="",
                   iter=nothing, system_name=vtkname(system),
                   custom_quantities...)
    mkpath(output_directory)

    # handle "_" on optional pre/postfix strings
    add_opt_str_pre(str) = (str === "" ? "" : "$(str)_")
    add_opt_str_post(str) = (str === nothing ? "" : "_$(str)")

    file = joinpath(output_directory,
                    add_opt_str_pre(prefix) * "$system_name"
                    * add_opt_str_post(iter))

    points = periodic_coords(active_coordinates(u, system), periodic_box)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(file, points, cells) do vtk
        write2vtk!(vtk, v, u, t, system)

        # Store particle index
        vtk["index"] = eachparticle(system)

        # Extract custom quantities for this system
        for (key, func) in custom_quantities
            value = func(v, u, t, system)
            if value !== nothing
                vtk[string(key)] = func(v, u, t, system)
            end
        end
    end
end

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
vtkname(system::OpenBoundarySPHSystem) = "open_boundary"

function write2vtk!(vtk, v, u, t, system::FluidSystem)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in active_particles(system)]
    vtk["density"] = [particle_density(v, system, particle)
                      for particle in active_particles(system)]
    vtk["pressure"] = [particle_pressure(v, system, particle)
                       for particle in active_particles(system)]

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
