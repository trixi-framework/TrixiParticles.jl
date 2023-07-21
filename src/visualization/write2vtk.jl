function trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="",
                   custom_quantities...)
    @unpack systems, neighborhood_searches = semi
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

    collection_file = joinpath(output_directory,
                    add_opt_str_pre(prefix) * "$system_name")

    pvd = paraview_collection(collection_file; append = true)

    points = periodic_coords(current_coordinates(u, system), periodic_box)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]


    vtk_grid(file, points, cells) do vtk
        write2vtk!(vtk, v, u, t, system)

        # Store particle index
        vtk["index"] = eachparticle(system)

        # Write some meta data
        vtk["time"] = t

        # Extract custom quantities for this system
        for (key, func) in custom_quantities
            value = func(v, u, t, system)
            if value !== nothing
                vtk[string(key)] = func(v, u, t, system)
            end
        end

        # add to collection
        pvd[t] = vtk
    end
    vtk_save(pvd)
end

function trixi2vtk(coordinates; output_directory="out", prefix="", filename="coordinates")
    mkpath(output_directory)
    file = prefix === "" ? joinpath(output_directory, filename) :
           joinpath(output_directory, "$(prefix)_$filename")

    points = coordinates
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]
    vtk_grid(vtk -> nothing, file, points, cells)

    return file
end

vtkname(system::WeaklyCompressibleSPHSystem) = "fluid"
vtkname(system::TotalLagrangianSPHSystem) = "solid"
vtkname(system::BoundarySPHSystem) = "boundary"

function write2vtk!(vtk, v, u, t, system::WeaklyCompressibleSPHSystem)
    @unpack density_calculator, cache = system

    vtk["velocity"] = view(v, 1:ndims(system), :)
    vtk["density"] = [particle_density(v, system, particle)
                      for particle in eachparticle(system)]
    vtk["pressure"] = system.pressure

    # write meta data
    vtk["solver"] = "WCSPH"
    vtk["correction_method"] = type2string(system.correction)
    vtk["acceleration"] = system.acceleration
    vtk["viscosity"] = type2string(system.viscosity)
    vtk["smoothing_kernel"] = type2string(system.smoothing_kernel)
    vtk["smoothing_length"] = system.smoothing_length
    vtk["density_calculator"] = type2string(system.density_calculator)
    vtk["state_equation"] = type2string(system.state_equation)

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
    @unpack boundary_model = system

    # write meta data
    vtk["boundary_model"] = "BoundaryModelDummyParticles"
    vtk["smoothing_kernel"] = type2string(boundary_model.smoothing_kernel)
    vtk["smoothing_length"] = boundary_model.smoothing_length
    vtk["density_calculator"] = type2string(boundary_model.density_calculator)
    vtk["state_equation"] = type2string(boundary_model.state_equation)

    write2vtk!(vtk, v, u, t, model, model.viscosity, system)
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles, viscosity, system)
    vtk["hydrodynamic_density"] = [particle_density(v, system, particle)
                                   for particle in eachparticle(system)]
    vtk["pressure"] = model.pressure

    # write meta data
    vtk["viscosity_model"] = type2string(viscosity)
    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles,
                    viscosity::ViscosityAdami, system)
    vtk["hydrodynamic_density"] = [particle_density(v, system, particle)
                                   for particle in eachparticle(system)]
    vtk["pressure"] = model.pressure
    vtk["wall_velocity"] = view(model.cache.wall_velocity, 1:ndims(system), :)

    # write meta data
    vtk["viscosity_model"] = "ViscosityAdami"

    return vtk
end
