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
              max_coordinates=Inf, custom_quantities...)

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
function trixi2vtk(vu_ode, semi, t; iter=nothing, output_directory="out",
                   prefix="", git_hash=compute_git_hash(), max_coordinates=Inf,
                   custom_quantities...)

    # The first argument is not necessary in most cases. Since it is usually not available to the user,
    # this API wrapper makes it optional.
    # Note that custom quantities using the fluid acceleration will not work and return NaN acceleration.
    return trixi2vtk(fill!(similar(vu_ode), NaN), vu_ode, semi, t; iter, output_directory,
                     prefix, git_hash, max_coordinates, custom_quantities...)
end

function trixi2vtk(dvdu_ode, vu_ode, semi, t; iter=nothing, output_directory="out",
                   prefix="", git_hash=compute_git_hash(), max_coordinates=Inf,
                   custom_quantities...)
    (; systems) = semi

    # Update quantities that are stored in the systems. These quantities (e.g. pressure)
    # still have the values from the last stage of the previous step if not updated here.
    @trixi_timeit timer() "update systems" begin
        v_ode, u_ode = vu_ode.x
        # Don't create sub-timers here to avoid cluttering the timer output
        @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)
    end

    filenames = system_names(systems)

    foreach_system(semi) do system
        system_index = system_indices(system, semi)
        periodic_box = get_neighborhood_search(system, semi).periodic_box

        trixi2vtk(system, dvdu_ode, vu_ode, semi, t, periodic_box;
                  system_name=filenames[system_index], output_directory, iter, prefix,
                  git_hash, max_coordinates, custom_quantities...)
    end
end

# Convert data for a single TrixiParticle system to VTK format
function trixi2vtk(system_, dvdu_ode_, vu_ode_, semi_, t, periodic_box;
                   output_directory="out", prefix="", iter=nothing,
                   system_name=vtkname(system_), max_coordinates=Inf,
                   git_hash=compute_git_hash(), custom_quantities...)
    mkpath(output_directory)

    # Skip empty systems
    if nparticles(system_) == 0
        return
    end

    v_ode_, u_ode_ = vu_ode_.x

    # Transfer to CPU if data is on the GPU. Do nothing if already on CPU.
    v_ode, u_ode, system, semi = transfer2cpu(v_ode_, u_ode_, system_, semi_)

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    file = joinpath(output_directory,
                    add_underscore_to_optional_prefix(prefix) * "$system_name"
                    * add_underscore_to_optional_postfix(iter))

    collection_file = joinpath(output_directory,
                               add_underscore_to_optional_prefix(prefix) * "$system_name")

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
        # Dispatches based on the different system types e.g. AbstractFluidSystem
        write2vtk!(vtk, v, u, t, system)

        # Store particle index
        vtk["index"] = eachparticle(system)
        vtk["time"] = t
        vtk["ndims"] = ndims(system)

        vtk["particle_spacing"] = [particle_spacing(system, particle)
                                   for particle in each_active_particle(system)]

        # Extract custom quantities for this system
        if !isempty(custom_quantities)
            dv_ode_, du_ode_ = dvdu_ode_.x
            dv_ode, du_ode = transfer2cpu(dv_ode_, du_ode_)

            for (key, quantity) in custom_quantities
                value = custom_quantity(quantity, system, dv_ode, du_ode, v_ode, u_ode,
                                        semi, t)
                if value !== nothing
                    vtk[string(key)] = value
                end
            end
        end

        # Add to collection
        pvd[t] = vtk
    end
    vtk_save(pvd)
end

function transfer2cpu(v_::AbstractGPUArray, u_, system_, semi_)
    semi = Adapt.adapt(Array, semi_)
    system_index = system_indices(system_, semi_)
    system = semi.systems[system_index]

    v, u = transfer2cpu(v_, u_)

    return v, u, system, semi
end

function transfer2cpu(v_, u_, system_, semi_)
    return v_, u_, system_, semi_
end

function transfer2cpu(v_::AbstractGPUArray, u_)
    v = transfer2cpu(v_)
    u = transfer2cpu(u_)

    return v, u
end

function transfer2cpu(v_, u_)
    return v_, u_
end

function transfer2cpu(a_::AbstractGPUArray)
    return Adapt.adapt(Array, a_)
end

function transfer2cpu(a_)
    return a_
end

function custom_quantity(quantity::AbstractArray, system, dv_ode, du_ode, v_ode, u_ode,
                         semi, t)
    return quantity
end

function custom_quantity(quantity, system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    # Check if `quantity` is a function of `system`, `v_ode`, `u_ode`, `semi` and `t`
    if !isempty(methods(quantity,
                        (typeof(system), typeof(dv_ode), typeof(du_ode), typeof(v_ode),
                         typeof(u_ode), typeof(semi), typeof(t))))
        return quantity(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    end

    # Assume `quantity` is a function of `data`
    data = system_data(system, dv_ode, du_ode, v_ode, u_ode, semi)
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

function write2vtk!(vtk, v, u, t, system)
    vtk["velocity"] = view(v, 1:ndims(system), :)

    return vtk
end

function write2vtk!(vtk, v, u, t, system::DEMSystem)
    vtk["velocity"] = view(v, 1:ndims(system), :)
    vtk["mass"] = [hydrodynamic_mass(system, particle)
                   for particle in eachparticle(system)]
    vtk["radius"] = [particle_radius(system, particle)
                     for particle in eachparticle(system)]
    return vtk
end

function write2vtk!(vtk, v, u, t, system::AbstractFluidSystem)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in eachparticle(system)]
    vtk["density"] = [current_density(v, system, particle)
                      for particle in eachparticle(system)]
    # Indexing the pressure is a workaround for slicing issue (see https://github.com/JuliaSIMD/StrideArrays.jl/issues/88)
    vtk["pressure"] = [current_pressure(v, system, particle)
                       for particle in eachparticle(system)]

    if system.surface_normal_method !== nothing
        vtk["surf_normal"] = [surface_normal(system, particle)
                              for particle in eachparticle(system)]
        vtk["neighbor_count"] = system.cache.neighbor_count
        vtk["color"] = system.cache.color
    end

    if system.surface_tension isa SurfaceTensionMorris ||
       system.surface_tension isa SurfaceTensionMomentumMorris
        surface_tension = zeros((ndims(system), n_integrated_particles(system)))
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
                                                                system,
                                                                system,
                                                                particle,
                                                                neighbor,
                                                                pos_diff,
                                                                distance,
                                                                rho_a,
                                                                rho_b,
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

function write2vtk!(vtk, v, u, t, system::TotalLagrangianSPHSystem)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in eachparticle(system)]
    vtk["jacobian"] = [det(deformation_gradient(system, particle))
                       for particle in eachparticle(system)]

    vtk["von_mises_stress"] = von_mises_stress(system)

    vtk["displacement"] = [current_coords(system, particle) -
                           initial_coords(system, particle)
                           for particle in eachparticle(system)]

    vtk["is_clamped"] = vcat(fill(0, system.n_integrated_particles),
                             fill(1, nparticles(system) - system.n_integrated_particles))

    vtk["lame_lambda"] = system.lame_lambda
    vtk["lame_mu"] = system.lame_mu
    vtk["young_modulus"] = system.young_modulus
    vtk["poisson_ratio"] = system.poisson_ratio

    sigma = cauchy_stress(system)
    vtk["sigma_11"] = sigma[1, 1, :]
    vtk["sigma_22"] = sigma[2, 2, :]
    if ndims(system) == 3
        vtk["sigma_33"] = sigma[3, 3, :]
    end

    vtk["material_density"] = system.material_density

    write2vtk!(vtk, v, u, t, system.boundary_model, system)
end

function write2vtk!(vtk, v, u, t, system::OpenBoundarySystem)
    vtk["velocity"] = [current_velocity(v, system, particle)
                       for particle in eachparticle(system)]
    vtk["density"] = [current_density(v, system, particle)
                      for particle in eachparticle(system)]
    vtk["pressure"] = [current_pressure(v, system, particle)
                       for particle in eachparticle(system)]

    return vtk
end

function write2vtk!(vtk, v, u, t, system::WallBoundarySystem)
    write2vtk!(vtk, v, u, t, system.boundary_model, system)
end

function write2vtk!(vtk, v, u, t, model::Nothing, system)
    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelMonaghanKajtar, system)
    return vtk
end

function write2vtk!(vtk, v, u, t, model::BoundaryModelDummyParticles, system)
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

function write2vtk!(vtk, v, u, t, system::BoundaryDEMSystem)
    return vtk
end
