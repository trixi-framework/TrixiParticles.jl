function trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="",
                   custom_quantities...)
    @unpack particle_containers = semi
    v_ode, u_ode = vu_ode.x

    # Add `_i` to each container name, where `i` is the index of the corresponding
    # container type.
    # `["fluid", "boundary", "boundary"]` becomes `["fluid_1", "boundary_1", "boundary_2"]`.
    cnames = particle_containers .|> vtkname
    filenames = [string(cnames[i], "_", count(==(cnames[i]), cnames[1:i]))
                 for i in eachindex(cnames)]

    foreach_enumerate(particle_containers) do (container_index, container)
        v = wrap_v(v_ode, container_index, container, semi)
        u = wrap_u(u_ode, container_index, container, semi)
        trixi2vtk(v, u, t, container; output_directory=output_directory,
                  container_name=filenames[container_index], iter=iter, prefix=prefix,
                  custom_quantities...)
    end
end

function trixi2vtk(v, u, t, container; output_directory="out", prefix="", iter=nothing,
                   container_name=vtkname(container),
                   custom_quantities...)
    mkpath(output_directory)

    # handle "_" on optional pre/postfix strings
    add_opt_str_pre(str) = (str === "" ? "" : "$(str)_")
    add_opt_str_post(str) = (str === nothing ? "" : "_$(str)")

    file = joinpath(output_directory,
                    add_opt_str_pre(prefix) * "$container_name"
                    * add_opt_str_post(iter))

    points = current_coordinates(u, container)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(file, points, cells) do vtk
        write2vtk!(vtk, v, u, t, container)

        # Store particle index
        vtk["index"] = eachparticle(container)

        # Extract custom quantities for this container
        for (key, func) in custom_quantities
            value = func(v, u, t, container)
            if value !== nothing
                vtk[string(key)] = func(v, u, t, container)
            end
        end
    end
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

vtkname(container::FluidParticleContainer) = "fluid"
vtkname(container::SolidParticleContainer) = "solid"
vtkname(container::BoundaryParticleContainer) = "boundary"

function write2vtk!(vtk, v, u, t, container::FluidParticleContainer)
    @unpack density_calculator, cache = container

    vtk["velocity"] = view(v, 1:ndims(container), :)
    vtk["density"] = [particle_density(v, container, particle)
                      for particle in eachparticle(container)]
    vtk["pressure"] = container.pressure

    return vtk
end

function write2vtk!(vtk, v, u, t, container::SolidParticleContainer)
    n_fixed_particles = nparticles(container) - n_moving_particles(container)

    vtk["velocity"] = hcat(view(v, 1:ndims(container), :),
                           zeros(ndims(container), n_fixed_particles))
    vtk["material_density"] = container.material_density

    write2vtk!(vtk, v, u, t, container.boundary_model, container)
end

function write2vtk!(vtk, v, u, t, container::BoundaryParticleContainer)
    write2vtk!(vtk, v, u, t, container.boundary_model, container)
end

function write2vtk!(vtk, v, u, t, model, container)
    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles, container)
    vtk["hydrodynamic_density"] = [particle_density(v, container, particle)
                                   for particle in eachparticle(container)]
    vtk["pressure"] = model.pressure

    return vtk
end
