function trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out",
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
                  container_name=filenames[container_index], iter=iter,
                  custom_quantities...)
    end
end

function trixi2vtk(v, u, t, container; output_directory="out", iter=nothing,
                   container_name=vtkname(container),
                   custom_quantities...)
    mkpath(output_directory)

    if iter === nothing
        file = joinpath(output_directory, "$container_name")
    else
        file = joinpath(output_directory, "$(container_name)_$iter")
    end

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

function trixi2vtk(coordinates; output_directory="out", filename="coordinates")
    mkpath(output_directory)
    file = joinpath(output_directory, filename)

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
    vtk["pressure"] = container.pressure

    write2vtk!(vtk, v, density_calculator, container)
end

function write2vtk!(vtk, v, ::SummationDensity, container::FluidParticleContainer)
    vtk["density"] = container.cache.density

    return vtk
end

function write2vtk!(vtk, v, ::ContinuityDensity, container::FluidParticleContainer)
    vtk["density"] = view(v, ndims(container) + 1, :)

    return vtk
end

function write2vtk!(vtk, v, u, t, container::SolidParticleContainer)
    n_fixed_particles = nparticles(container) - n_moving_particles(container)

    vtk["velocity"] = hcat(view(v, 1:ndims(container), :),
                           zeros(ndims(container), n_fixed_particles))
    vtk["material_density"] = container.material_density

    return vtk
end

function write2vtk!(vtk, v, u, t, container::BoundaryParticleContainer)
    write2vtk!(vtk, v, u, t, container.boundary_model, container)
end

function write2vtk!(vtk, v, u, t, model, container::BoundaryParticleContainer)
    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles,
                    container::BoundaryParticleContainer)
    vtk["density"] = [get_particle_density(particle, v, container)
                      for particle in eachparticle(container)]
    vtk["pressure"] = model.pressure

    return vtk
end
