"""
    trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="",
              write_meta_data=true, custom_quantities...)

Convert Trixi simulation data to VTK format.

# Arguments
- `vu_ode`: Solution of the TrixiParticles ODE system at one time step. This expects an `ArrayPartition` as returned in the examples as `sol`.
- `semi`:   Semidiscretization of the TrixiParticles simulation.
- `t`:      Current time of the simulation.

# Keywords
- `iter`:                 Iteration number when multiple iterations are to be stored in separate files.
- `output_directory`:     Output directory path. Defaults to `"out"`.
- `prefix`:               Prefix for output files. Defaults to an empty string.
- `write_meta_data`:      Write meta data.
- `custom_quantities...`: Additional custom quantities to include in the VTK output. TODO.


# Example
```julia
trixi2vtk(sol.u[end], semi, 0.0, iter=1, output_directory="output", prefix="solution")

TODO: example for custom_quantities
"""
function trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="",
                   write_meta_data=true, custom_quantities...)
    (; systems) = semi
    v_ode, u_ode = vu_ode.x

    filenames = system_names(systems)

    foreach_system(semi) do system
        system_index = system_indices(system, semi)

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        periodic_box = neighborhood_searches(system, system, semi).periodic_box

        trixi2vtk(v, u, t, system, periodic_box;
                  output_directory=output_directory,
                  system_name=filenames[system_index], iter=iter, prefix=prefix,
                  write_meta_data=write_meta_data, custom_quantities...)
    end
end

# Convert data for a single TrixiParticle system to VTK format
function trixi2vtk(v, u, t, system, periodic_box; output_directory="out", prefix="",
                   iter=nothing, system_name=vtkname(system), write_meta_data=true,
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

    # Reset the collection when the iteration is 0
    pvd = paraview_collection(collection_file; append=iter > 0)

    points = periodic_coords(current_coordinates(u, system), periodic_box)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(file, points, cells) do vtk
        write2vtk!(vtk, v, u, t, system, write_meta_data=write_meta_data)

        # Store particle index
        vtk["index"] = eachparticle(system)
        vtk["time"] = t

        # Extract custom quantities for this system
        for (key, quantity) in custom_quantities
            value = custom_quantity(quantity, v, u, t, system)
            if value !== nothing
                vtk[string(key)] = value
            end
        end

        # Add to collection
        pvd[t] = vtk
    end
    vtk_save(pvd)
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

function write2vtk!(vtk, v, u, t, system::FluidSystem; write_meta_data=true)
    vtk["velocity"] = view(v, 1:ndims(system), :)
    vtk["density"] = [particle_density(v, system, particle)
                      for particle in eachparticle(system)]
    vtk["pressure"] = [particle_pressure(v, system, particle)
                       for particle in eachparticle(system)]

    if write_meta_data
        vtk["acceleration"] = system.acceleration
        vtk["viscosity"] = type2string(system.viscosity)
        write2vtk!(vtk, system.viscosity)
        vtk["smoothing_kernel"] = type2string(system.smoothing_kernel)
        vtk["smoothing_length"] = system.smoothing_length
        vtk["density_calculator"] = type2string(system.density_calculator)

        if system isa WeaklyCompressibleSPHSystem
            vtk["correction_method"] = type2string(system.correction)
            if system.correction isa AkinciFreeSurfaceCorrection
                vtk["correction_rho0"] = system.correction.rho0
            end
            vtk["state_equation"] = type2string(system.state_equation)
            vtk["state_equation_rho0"] = system.state_equation.reference_density
            vtk["state_equation_p0"] = system.state_equation.reference_pressure
            vtk["state_equation_pa"] = system.state_equation.background_pressure
            vtk["state_equation_c"] = system.state_equation.sound_speed
            if system.state_equation isa StateEquationCole
                vtk["state_equation_gamma"] = system.state_equation.gamma
            end

            vtk["solver"] = "WCSPH"
        else
            vtk["solver"] = "EDAC"
            vtk["sound_speed"] = system.sound_speed
        end
    end

    return vtk
end

write2vtk!(vtk, viscosity::NoViscosity) = vtk

function write2vtk!(vtk, viscosity::ViscosityAdami)
    vtk["viscosity_nu"] = viscosity.nu
    vtk["viscosity_epsilon"] = viscosity.epsilon
end

function write2vtk!(vtk, viscosity::ArtificialViscosityMonaghan)
    vtk["viscosity_alpha"] = viscosity.alpha
    vtk["viscosity_beta"] = viscosity.beta
    vtk["viscosity_epsilon"] = viscosity.epsilon
end

function write2vtk!(vtk, v, u, t, system::TotalLagrangianSPHSystem; write_meta_data=true)
    n_fixed_particles = nparticles(system) - n_moving_particles(system)

    vtk["velocity"] = hcat(view(v, 1:ndims(system), :),
                           zeros(ndims(system), n_fixed_particles))
    vtk["material_density"] = system.material_density
    vtk["young_modulus"] = system.young_modulus
    vtk["poisson_ratio"] = system.poisson_ratio
    vtk["lame_lambda"] = system.lame_lambda
    vtk["lame_mu"] = system.lame_mu

    write2vtk!(vtk, v, u, t, system.boundary_model, system, write_meta_data=write_meta_data)
end

function write2vtk!(vtk, v, u, t, system::BoundarySPHSystem; write_meta_data=true)
    write2vtk!(vtk, v, u, t, system.boundary_model, system, write_meta_data=write_meta_data)
end

function write2vtk!(vtk, v, u, t, model, system; write_meta_data=true)
    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelMonaghanKajtar, system;
                    write_meta_data=true)
    if write_meta_data
        vtk["boundary_model"] = "BoundaryModelMonaghanKajtar"
        vtk["boundary_spacing_ratio"] = model.beta
        vtk["boundary_K"] = model.K
    end
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles, system;
                    write_meta_data=true)
    if write_meta_data
        vtk["boundary_model"] = "BoundaryModelDummyParticles"
        vtk["smoothing_kernel"] = type2string(system.boundary_model.smoothing_kernel)
        vtk["smoothing_length"] = system.boundary_model.smoothing_length
        vtk["density_calculator"] = type2string(system.boundary_model.density_calculator)
        vtk["state_equation"] = type2string(system.boundary_model.state_equation)
    end

    write2vtk!(vtk, v, u, t, model, model.viscosity, system,
               write_meta_data=write_meta_data)
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles, viscosity, system;
                    write_meta_data=true)
    vtk["hydrodynamic_density"] = [particle_density(v, system, particle)
                                   for particle in eachparticle(system)]
    vtk["pressure"] = model.pressure

    if write_meta_data
        vtk["viscosity_model"] = type2string(viscosity)
    end

    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles,
                    viscosity::ViscosityAdami, system; write_meta_data=true)
    vtk["hydrodynamic_density"] = [particle_density(v, system, particle)
                                   for particle in eachparticle(system)]
    vtk["pressure"] = model.pressure
    vtk["wall_velocity"] = view(model.cache.wall_velocity, 1:ndims(system), :)

    if write_meta_data
        vtk["viscosity_model"] = "ViscosityAdami"
    end

    return vtk
end
