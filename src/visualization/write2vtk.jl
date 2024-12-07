function system_names(systems)
    # Add `_i` to each system name, where `i` is the index of the corresponding
    # system type.
    # `["fluid", "boundary", "boundary"]` becomes `["fluid_1", "boundary_1", "boundary_2"]`.
    cnames = vtkname.(systems)
    filenames = [string(cnames[i], "_", count(==(cnames[i]), cnames[1:i]))
                 for i in eachindex(cnames)]
    return filenames
end

"""
    trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="",
              write_meta_data=true, max_coordinates=Inf, custom_quantities...)

Convert Trixi simulation data to VTK format.

# Arguments
- `vu_ode`: Solution of the TrixiParticles ODE system at one time step.
            This expects an `ArrayPartition` as returned in the examples as `sol.u[end]`.
- `semi`:   Semidiscretization of the TrixiParticles simulation.
- `t`:      Current time of the simulation.

# Keywords
- `iter=nothing`:           Iteration number when multiple iterations are to be stored in
                            separate files. This number is just appended to the filename.
- `output_directory="out"`: Output directory path.
- `prefix=""`:              Prefix for output files.
- `write_meta_data=true`:   Write meta data.
- `max_coordinates=Inf`     The coordinates of particles will be clipped if their absolute
                            values exceed this threshold.
- `custom_quantities...`:   Additional custom quantities to include in the VTK output.
                            Each custom quantity must be a function of `(v, u, t, system)`,
                            which will be called for every system, where `v` and `u` are the
                            wrapped solution arrays for the corresponding system and `t` is
                            the current simulation time. Note that working with these `v`
                            and `u` arrays requires undocumented internal functions of
                            TrixiParticles. See [Custom Quantities](@ref custom_quantities)
                            for a list of pre-defined custom quantities that can be used here.

# Example
```jldoctest; output = false, setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing))
trixi2vtk(sol.u[end], semi, 0.0, iter=1, output_directory="output", prefix="solution")

# Additionally store the kinetic energy of each system as "my_custom_quantity"
trixi2vtk(sol.u[end], semi, 0.0, iter=1, my_custom_quantity=kinetic_energy)

# output

```
"""
function trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out", prefix="",
                   write_meta_data=true, git_hash=compute_git_hash(),
                   max_coordinates=Inf, custom_quantities...)
    (; systems) = semi
    v_ode, u_ode = vu_ode.x

    # Update quantities that are stored in the systems. These quantities (e.g. pressure)
    # still have the values from the last stage of the previous step if not updated here.
    @trixi_timeit timer() "update systems" update_systems_and_nhs(v_ode, u_ode, semi, t;
                                                                  update_from_callback=true)

    filenames = system_names(systems)

    foreach_system(semi) do system
        system_index = system_indices(system, semi)

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        periodic_box = get_neighborhood_search(system, semi).periodic_box

        trixi2vtk(v, u, t, system, periodic_box;
                  system_name=filenames[system_index], output_directory, iter, prefix,
                  write_meta_data, git_hash, max_coordinates, custom_quantities...)
    end
end

# Convert data for a single TrixiParticle system to VTK format
function trixi2vtk(v_, u_, t, system_, periodic_box; output_directory="out", prefix="",
                   iter=nothing, system_name=vtkname(system_), write_meta_data=true,
                   max_coordinates=Inf, git_hash=compute_git_hash(),
                   custom_quantities...)
    mkpath(output_directory)

    # Skip empty systems
    if nparticles(system_) == 0
        return
    end

    # Transfer to CPU if data is on the GPU. Do nothing if already on CPU.
    v, u, system = transfer2cpu(v_, u_, system_)

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

    points = PointNeighbors.periodic_coords(active_coordinates(u, system),
                                            periodic_box)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    if abs(maximum(points)) > max_coordinates || abs(minimum(points)) > max_coordinates
        println("Warning: At least one particle's absolute coordinates exceed `max_coordinates`"
                *
                " and have been clipped")
        for i in eachindex(points)
            points[i] = clamp(points[i], -max_coordinates, max_coordinates)
        end
    end

    @trixi_timeit timer() "write to vtk" vtk_grid(file, points, cells) do vtk
        # dispatches based on the different system types e.g. FluidSystem, TotalLagrangianSPHSystem
        write2vtk!(vtk, v, u, t, system, write_meta_data=write_meta_data)

        # Store particle index
        vtk["index"] = active_particles(system)
        vtk["time"] = t

        if write_meta_data
            vtk["solver_version"] = git_hash
            vtk["julia_version"] = string(VERSION)
        end

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

function transfer2cpu(v_, u_, system_::GPUSystem)
    v = Adapt.adapt(Array, v_)
    u = Adapt.adapt(Array, u_)
    system = Adapt.adapt(Array, system_)

    return v, u, system
end

function transfer2cpu(v_, u_, system_)
    return v_, u_, system_
end

function custom_quantity(quantity::AbstractArray, v, u, t, system)
    return quantity
end

function custom_quantity(quantity, v, u, t, system)
    # Assume `quantity` is a function of `v`, `u`, `t`, and `system`
    return quantity(v, u, t, system)
end

"""
    trixi2vtk(coordinates; output_directory="out", prefix="", filename="coordinates",
              custom_quantities...)

Convert coordinate data to VTK format.

# Arguments
- `coordinates`: Coordinates to be saved.

# Keywords
- `output_directory="out"`: Output directory path.
- `prefix=""`:              Prefix for the output file.
- `filename="coordinates"`: Name of the output file.
- `custom_quantities...`:   Additional custom quantities to include in the VTK output.

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

"""
    trixi2vtk(initial_condition::InitialCondition; output_directory="out",
              prefix="", filename="initial_condition", custom_quantities...)

Convert [`InitialCondition`](@ref) data to VTK format.

# Arguments
- `initial_condition`: [`InitialCondition`](@ref) to be saved.

# Keywords
- `output_directory="out"`: Output directory path.
- `prefix=""`:              Prefix for the output file.
- `filename="coordinates"`: Name of the output file.
- `custom_quantities...`:   Additional custom quantities to include in the VTK output.

# Returns
- `file::AbstractString`: Path to the generated VTK file.
"""
function trixi2vtk(initial_condition::InitialCondition; output_directory="out",
                   prefix="", filename="initial_condition", custom_quantities...)
    (; coordinates, velocity, density, mass, pressure) = initial_condition

    return trixi2vtk(coordinates; output_directory, prefix, filename,
                     density=density, initial_velocity=velocity, mass=mass,
                     pressure=pressure, custom_quantities...)
end

function write2vtk!(vtk, v, u, t, system; write_meta_data=true)
    vtk["velocity"] = view(v, 1:ndims(system), :)

    return vtk
end

function write2vtk!(vtk, v, u, t, system::FluidSystem; write_meta_data=true)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in active_particles(system)]
    vtk["density"] = [particle_density(v, system, particle)
                      for particle in active_particles(system)]
    vtk["pressure"] = [particle_pressure(v, system, particle)
                       for particle in active_particles(system)]

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

            if system.state_equation isa StateEquationCole
                vtk["state_equation_exponent"] = system.state_equation.exponent
            end

            if system.state_equation isa StateEquationIdealGas
                vtk["state_equation_gamma"] = system.state_equation.gamma
            end

            vtk["state_equation"] = type2string(system.state_equation)
            vtk["state_equation_rho0"] = system.state_equation.reference_density
            vtk["state_equation_pa"] = system.state_equation.background_pressure
            vtk["state_equation_c"] = system.state_equation.sound_speed
            vtk["solver"] = "WCSPH"
        else
            vtk["solver"] = "EDAC"
            vtk["sound_speed"] = system.sound_speed
            vtk["background_pressure_TVF"] = system.transport_velocity isa Nothing ?
                                             "-" :
                                             system.transport_velocity.background_pressure
        end
    end

    return vtk
end

write2vtk!(vtk, viscosity::Nothing) = vtk

function write2vtk!(vtk, viscosity::Union{ViscosityAdami, ViscosityMorris})
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
    vtk["jacobian"] = [det(deformation_gradient(system, particle))
                       for particle in eachparticle(system)]

    vtk["von_mises_stress"] = von_mises_stress(system)

    sigma = cauchy_stress(system)
    vtk["sigma_11"] = sigma[1, 1, :]
    vtk["sigma_22"] = sigma[2, 2, :]
    if ndims(system) == 3
        vtk["sigma_33"] = sigma[3, 3, :]
    end

    vtk["material_density"] = system.material_density

    if write_meta_data
        vtk["young_modulus"] = system.young_modulus
        vtk["poisson_ratio"] = system.poisson_ratio
        vtk["lame_lambda"] = system.lame_lambda
        vtk["lame_mu"] = system.lame_mu
        vtk["smoothing_kernel"] = type2string(system.smoothing_kernel)
        vtk["smoothing_length"] = system.smoothing_length
    end

    write2vtk!(vtk, v, u, t, system.boundary_model, system, write_meta_data=write_meta_data)
end

function write2vtk!(vtk, v, u, t, system::OpenBoundarySPHSystem; write_meta_data=true)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in active_particles(system)]
    vtk["density"] = [particle_density(v, system, particle)
                      for particle in active_particles(system)]
    vtk["pressure"] = [particle_pressure(v, system, particle)
                       for particle in active_particles(system)]

    if write_meta_data
        vtk["boundary_zone"] = type2string(system.boundary_zone)
        vtk["width"] = round(system.boundary_zone.zone_width, digits=3)
        vtk["flow_direction"] = system.flow_direction
        vtk["velocity_function"] = type2string(system.reference_velocity)
        vtk["pressure_function"] = type2string(system.reference_pressure)
        vtk["density_function"] = type2string(system.reference_density)
    end

    return vtk
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
        vtk["viscosity_model"] = type2string(model.viscosity)
    end

    vtk["hydrodynamic_density"] = [particle_density(v, system, particle)
                                   for particle in eachparticle(system)]
    vtk["pressure"] = model.pressure

    if model.viscosity isa ViscosityAdami
        vtk["wall_velocity"] = view(model.cache.wall_velocity, 1:ndims(system), :)
    end
end

function write2vtk!(vtk, v, u, t, system::BoundaryDEMSystem; write_meta_data=true)
    return vtk
end
