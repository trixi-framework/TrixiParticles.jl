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
                            Each custom quantity must be a function of `(system, data, t)`,
                            which will be called for every system, where `data` is a named
                            tuple with fields depending on the system type, and `t` is the
                            current simulation time. Check the available data for each
                            system with `available_data(system)`.
                            See [Custom Quantities](@ref custom_quantities)
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
    @trixi_timeit timer() "update systems" begin
        # Don't create sub-timers here to avoid cluttering the timer output
        @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t;
                                                 update_from_callback=true)
    end

    filenames = system_names(systems)

    foreach_system(semi) do system
        system_index = system_indices(system, semi)
        periodic_box = get_neighborhood_search(system, semi).periodic_box

        trixi2vtk(system, v_ode, u_ode, semi, t, periodic_box;
                  system_name=filenames[system_index], output_directory, iter, prefix,
                  write_meta_data, git_hash, max_coordinates, custom_quantities...)
    end
end

# Convert data for a single TrixiParticle system to VTK format
function trixi2vtk(system_, v_ode_, u_ode_, semi_, t, periodic_box; output_directory="out",
                   prefix="", iter=nothing, system_name=vtkname(system_),
                   write_meta_data=true, max_coordinates=Inf, git_hash=compute_git_hash(),
                   custom_quantities...)
    mkpath(output_directory)

    # Skip empty systems
    if nparticles(system_) == 0
        return
    end

    # Transfer to CPU if data is on the GPU. Do nothing if already on CPU.
    v_ode, u_ode, system, semi = transfer2cpu(v_ode_, u_ode_, system_, semi_)

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

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
        # Dispatches based on the different system types e.g. FluidSystem, TotalLagrangianSPHSystem
        write2vtk!(vtk, v, u, t, system, write_meta_data=write_meta_data)

        # Store particle index
        vtk["index"] = active_particles(system)
        vtk["time"] = t
        vtk["ndims"] = ndims(system)

        vtk["particle_spacing"] = [particle_spacing(system, particle)
                                   for particle in active_particles(system)]

        if write_meta_data
            vtk["solver_version"] = git_hash
            vtk["julia_version"] = string(VERSION)
        end

        # Extract custom quantities for this system
        for (key, quantity) in custom_quantities
            value = custom_quantity(quantity, system, v_ode, u_ode, semi, t)
            if value !== nothing
                vtk[string(key)] = value
            end
        end

        # Add to collection
        pvd[t] = vtk
    end
    vtk_save(pvd)
end

function transfer2cpu(v_::AbstractGPUArray, u_, system_, semi_)
    v = Adapt.adapt(Array, v_)
    u = Adapt.adapt(Array, u_)
    semi = Adapt.adapt(Array, semi_)
    system_index = system_indices(system_, semi_)
    system = semi.systems[system_index]

    return v, u, system, semi
end

function transfer2cpu(v_, u_, system_, semi_)
    return v_, u_, system_, semi_
end

function custom_quantity(quantity::AbstractArray, system, v_ode, u_ode, semi, t)
    return quantity
end

function custom_quantity(quantity, system, v_ode, u_ode, semi, t)
    # Check if `quantity` is a function of `system`, `v_ode`, `u_ode`, `semi` and `t`
    if !isempty(methods(quantity,
                        (typeof(system), typeof(v_ode), typeof(u_ode),
                         typeof(semi), typeof(t))))
        return quantity(system, v_ode, u_ode, semi, t)
    end

    # Assume `quantity` is a function of `data`
    data = system_data(system, v_ode, u_ode, semi)
    return quantity(system, data, t)
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
                   particle_spacing=(-ones(size(coordinates, 2))), custom_quantities...)
    mkpath(output_directory)
    file = prefix === "" ? joinpath(output_directory, filename) :
           joinpath(output_directory, "$(prefix)_$filename")

    points = coordinates
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(file, points, cells) do vtk
        # Store particle index.
        vtk["index"] = [i for i in axes(coordinates, 2)]
        vtk["ndims"] = size(coordinates, 1)
        vtk["particle_spacing"] = particle_spacing

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
                     particle_spacing=(initial_condition.particle_spacing .*
                                       ones(nparticles(initial_condition))),
                     pressure=pressure, custom_quantities...)
end

function write2vtk!(vtk, v, u, t, system; write_meta_data=true)
    vtk["velocity"] = view(v, 1:ndims(system), :)

    return vtk
end

function write2vtk!(vtk, v, u, t, system::DEMSystem; write_meta_data=true)
    vtk["velocity"] = view(v, 1:ndims(system), :)
    vtk["mass"] = [hydrodynamic_mass(system, particle)
                   for particle in active_particles(system)]
    vtk["radius"] = [particle_radius(system, particle)
                     for particle in active_particles(system)]
    return vtk
end

function write2vtk!(vtk, v, u, t, system::FluidSystem; write_meta_data=true)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in active_particles(system)]
    vtk["density"] = [current_density(v, system, particle)
                      for particle in active_particles(system)]
    # Indexing the pressure is a workaround for slicing issue (see https://github.com/JuliaSIMD/StrideArrays.jl/issues/88)
    vtk["pressure"] = [current_pressure(v, system, particle)
                       for particle in active_particles(system)]

    if system.surface_normal_method !== nothing
        vtk["surf_normal"] = [surface_normal(system, particle)
                              for particle in eachparticle(system)]
        vtk["neighbor_count"] = system.cache.neighbor_count
        vtk["color"] = system.cache.color
    end

    if system.surface_tension isa SurfaceTensionMorris ||
       system.surface_tension isa SurfaceTensionMomentumMorris
        surface_tension = zeros((ndims(system), n_moving_particles(system)))
        system_coords = current_coordinates(u, system)

        surface_tension_a = surface_tension_model(system)
        surface_tension_b = surface_tension_model(system)
        nhs = create_neighborhood_search(nothing, system, system)

        foreach_point_neighbor(system_coords, system_coords,
                               nhs) do particle, neighbor, pos_diff, distance
            rho_a = current_density(v, system, particle)
            rho_b = current_density(v, system, neighbor)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            surface_tension[1:ndims(system),
                            particle] .+= surface_tension_force(surface_tension_a,
                                                                surface_tension_b,
                                                                system, system, particle,
                                                                neighbor, pos_diff,
                                                                distance, rho_a, rho_b,
                                                                grad_kernel)
        end
        vtk["surface_tension"] = surface_tension

        if system.surface_tension isa SurfaceTensionMorris
            vtk["curvature"] = system.cache.curvature
        end
        if system.surface_tension isa SurfaceTensionMomentumMorris
            vtk["surface_stress_tensor"] = system.cache.stress_tensor
        end
    end

    if write_meta_data
        vtk["acceleration"] = system.acceleration
        vtk["viscosity"] = type2string(system.viscosity)
        write2vtk!(vtk, system.viscosity)
        vtk["smoothing_kernel"] = type2string(system.smoothing_kernel)
        vtk["smoothing_length_factor"] = system.cache.smoothing_length_factor
        vtk["density_calculator"] = type2string(system.density_calculator)

        if system isa WeaklyCompressibleSPHSystem
            vtk["solver"] = "WCSPH"

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

function write2vtk!(vtk,
                    viscosity::Union{ViscosityAdami, ViscosityMorris, ViscosityAdamiSGS,
                                     ViscosityMorrisSGS})
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

    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in active_particles(system)]
    vtk["jacobian"] = [det(deformation_gradient(system, particle))
                       for particle in eachparticle(system)]

    vtk["von_mises_stress"] = von_mises_stress(system)

    vtk["displacement"] = [current_coords(system, particle) -
                           initial_coords(system, particle)
                           for particle in eachparticle(system)]

    sigma = cauchy_stress(system)
    vtk["sigma_11"] = sigma[1, 1, :]
    vtk["sigma_22"] = sigma[2, 2, :]
    if ndims(system) == 3
        vtk["sigma_33"] = sigma[3, 3, :]
    end

    vtk["material_density"] = system.material_density

    if write_meta_data
        vtk["lame_lambda"] = system.lame_lambda
        vtk["lame_mu"] = system.lame_mu
        vtk["smoothing_kernel"] = type2string(system.smoothing_kernel)
        vtk["smoothing_length_factor"] = initial_smoothing_length(system) /
                                         particle_spacing(system, 1)
    end

    write2vtk!(vtk, v, u, t, system.boundary_model, system, write_meta_data=write_meta_data)
end

function write2vtk!(vtk, v, u, t, system::OpenBoundarySPHSystem; write_meta_data=true)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in active_particles(system)]
    vtk["density"] = [current_density(v, system, particle)
                      for particle in active_particles(system)]
    vtk["pressure"] = [current_pressure(v, system, particle)
                       for particle in active_particles(system)]

    if write_meta_data
        vtk["boundary_zone"] = type2string(first(typeof(system.boundary_zone).parameters))
        vtk["width"] = round(system.boundary_zone.zone_width, digits=3)
        vtk["velocity_function"] = type2string(system.reference_velocity)
        vtk["pressure_function"] = type2string(system.reference_pressure)
        vtk["density_function"] = type2string(system.reference_density)
    end

    return vtk
end

function write2vtk!(vtk, v, u, t, system::BoundarySPHSystem; write_meta_data=true)
    write2vtk!(vtk, v, u, t, system.boundary_model, system, write_meta_data=write_meta_data)
end

function write2vtk!(vtk, v, u, t, model::Nothing, system; write_meta_data=true)
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
        vtk["smoothing_kernel"] = type2string(model.smoothing_kernel)
        vtk["smoothing_length"] = model.smoothing_length
        vtk["density_calculator"] = type2string(model.density_calculator)
        vtk["state_equation"] = type2string(model.state_equation)
        vtk["viscosity_model"] = type2string(model.viscosity)
    end

    vtk["hydrodynamic_density"] = current_density(v, system)
    vtk["pressure"] = model.pressure

    if haskey(model.cache, :initial_colorfield)
        vtk["initial_colorfield"] = model.cache.initial_colorfield
        vtk["colorfield"] = model.cache.colorfield
        vtk["neighbor_count"] = model.cache.neighbor_count
    end

    if model.viscosity isa ViscosityAdami
        vtk["wall_velocity"] = view(model.cache.wall_velocity, 1:ndims(system), :)
    end
end

function write2vtk!(vtk, v, u, t, system::BoundaryDEMSystem; write_meta_data=true)
    return vtk
end
