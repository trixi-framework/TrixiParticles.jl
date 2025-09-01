include("write_vtk.jl")
include("read_vtk.jl")

# handle "_" on optional prefix/filename strings
add_opt_str_post(str) = (str === "" ? "" : "_$(str)")

function write_meta_data(callback::SolutionSavingCallback, integrator)
    prefix = callback.prefix
    filename = callback.filename

    meta_data = create_meta_data_dict(callback, integrator)

    # write JSON-file
    output_directory = callback.output_directory
    mkpath(output_directory)
    json_file = joinpath(output_directory,
                         "meta_data" * add_opt_str_post(prefix) *
                         add_opt_str_post(filename) * ".json")

    open(json_file, "w") do file
        JSON.print(file, meta_data, 2)
    end
end

function create_meta_data_dict(callback, integrator)
    git_hash = callback.git_hash
    prefix = hasproperty(callback, :prefix) ? callback.prefix : ""
    semi = integrator.p
    names = system_names(semi.systems)

    meta_data = Dict{String, Any}()

    info = Dict{String, Any}()
    add_simulation_info!(info, git_hash, integrator)
    meta_data["simulation_info"] = info

    systems = Dict{String, Any}()
    foreach_system(semi) do system
        idx = system_indices(system, semi)
        name = prefix * add_opt_str_post("$(names[idx])")

        system_data = Dict{String, Any}()
        add_system_data!(system_data, system)

        systems[name] = system_data
    end
    meta_data["system_data"] = systems

    return meta_data
end

function add_simulation_info!(info, git_hash, integrator)
    info["solver_name"] = "TrixiParticles.jl"
    info["solver_version"] = git_hash[]
    info["julia_version"] = string(VERSION)

    info["time_integrator"] = Dict{String, Any}()
    info["time_integrator"]["integrator_type"] = type2string(integrator.alg)
    info["time_integrator"]["start_time"] = first(integrator.sol.prob.tspan)
    info["time_integrator"]["final_time"] = last(integrator.sol.prob.tspan)
    info["time_integrator"]["adaptive"] = integrator.opts.adaptive
    if integrator.opts.adaptive
        info["time_integrator"]["abstol"] = integrator.opts.abstol
        info["time_integrator"]["reltol"] = integrator.opts.reltol
        info["time_integrator"]["controller"] = type2string(integrator.opts.controller)
    end

    info["technical_setup"] = Dict{String, Any}()
    info["technical_setup"]["parallelization_backend"] = type2string(integrator.p.parallelization_backend)
    info["technical_setup"]["#threads"] = Threads.nthreads()
end

# Skip systemdata addition for `Nothing`
add_system_data!(system_data, data::Nothing) = system_data

function add_system_data!(system_data, data)
    throw(ArgumentError("Method for $(typeof(data)) not implemented. " *
                        "Please add a method `add_system_data!(system_data, ::$(typeof(data)))`."))
end

function add_system_data!(system_data, system::FluidSystem)
    system_data["system_type"] = type2string(system)
    system_data["particle_spacing"] = particle_spacing(system, 1)
    system_data["density_calculator"] = type2string(system.density_calculator)
    system_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    system_data["smoothing_length"] = system.cache.smoothing_length
    system_data["acceleration"] = system.acceleration
    system_data["sound_speed"] = system_sound_speed(system)
    system_data["pressure_acceleration_formulation"] = nameof(system.pressure_acceleration_formulation)
    add_system_data!(system_data, system.shifting_technique)
    add_system_data!(system_data, system.surface_tension)
    add_system_data!(system_data, system.surface_normal_method)
    add_system_data!(system_data, system.viscosity)
    add_system_data!(system_data, system.correction)
    add_system_data!(system_data, system_state_equation(system))
    if hasfield(typeof(system), :density_diffusion)
        add_system_data!(system_data, system.density_diffusion)
    end
    if hasfield(typeof(system), :alpha)
        system_data["alpha"] = system.alpha
    end
end

function add_system_data!(system_data, system::TotalLagrangianSPHSystem)
    system_data["system_type"] = type2string(system)
    system_data["particle_spacing"] = particle_spacing(system, 1)
    system_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    system_data["smoothing_length"] = system.smoothing_length
    system_data["acceleration"] = system.acceleration
    add_system_data!(system_data, system.boundary_model)
    add_system_data!(system_data, system.penalty_force)
end

function add_system_data!(system_data, system::BoundarySPHSystem)
    system_data["system_type"] = type2string(system)
    system_data["particle_spacing"] = particle_spacing(system, 1)
    system_data["adhesion_coefficient"] = system.adhesion_coefficient
    add_system_data!(system_data, system.boundary_model)
    add_system_data!(system_data, system.movement)
end

function add_system_data!(system_data, system::BoundaryDEMSystem)
    system_data["system_type"] = type2string(system)
    system_data["particle_spacing"] = particle_spacing(system, 1)
    system_data["normal_stiffness"] = system.normal_stiffness
end

function add_system_data!(system_data, system::DEMSystem)
    system_data["system_type"] = type2string(system)
    system_data["particle_spacing"] = particle_spacing(system, 1)
    system_data["damping_coefficient"] = system.damping_coefficient
    system_data["acceleration"] = system.acceleration
    add_system_data!(system_data, system.contact_model)
end

function add_system_data!(system_data, system::OpenBoundarySPHSystem)
    system_data["system_type"] = type2string(system)
    system_data["particle_spacing"] = particle_spacing(system, 1)
    system_data["reference_velocity"] = type2string(system.reference_velocity)
    system_data["reference_pressure"] = type2string(system.reference_pressure)
    system_data["reference_density"] = type2string(system.reference_density)
    add_system_data!(system_data, system.boundary_model)
    add_system_data!(system_data, system.boundary_zone)
end

function add_system_data!(system_data, system::ParticlePackingSystem)
    system_data["system_type"] = type2string(system)
    system_data["particle_spacing"] = system.particle_spacing
    system_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    system_data["smoothing_length_interpolation"] = system.smoothing_length_interpolation
    system_data["background_pressure"] = system.background_pressure
    system_data["place_on_shell"] = system.place_on_shell
    system_data["shift_length"] = system.shift_length
end

function add_system_data!(system_data, boundary_model::BoundaryModelDummyParticles)
    system_data["boundary_model"] = Dict{String, Any}()
    system_data["boundary_model"]["model"] = type2string(boundary_model)
    system_data["boundary_model"]["smoothing_kernel"] = type2string(boundary_model.smoothing_kernel)
    system_data["boundary_model"]["smoothing_length"] = boundary_model.smoothing_length
    system_data["boundary_model"]["density_calculator"] = type2string(boundary_model.density_calculator)
    add_system_data!(system_data["boundary_model"], boundary_model.state_equation)
    add_system_data!(system_data["boundary_model"], boundary_model.viscosity)
    add_system_data!(system_data["boundary_model"], boundary_model.correction)
end

function add_system_data!(system_data, boundary_model::BoundaryModelMonaghanKajtar)
    system_data["boundary_model"] = Dict{String, Any}()
    system_data["boundary_model"]["model"] = type2string(boundary_model)
    system_data["boundary_model"]["beta"] = boundary_model.beta
    system_data["boundary_model"]["K"] = boundary_model.K
    add_system_data!(system_data["boundary_model"], boundary_model.viscosity)
end

function add_system_data!(system_data, boundary_model::BoundaryModelTafuni)
    system_data["boundary_model"] = Dict{String, Any}()
    system_data["boundary_model"]["model"] = type2string(boundary_model)
end

function add_system_data!(system_data, boundary_model::BoundaryModelLastiwka)
    system_data["boundary_model"] = Dict{String, Any}()
    system_data["boundary_model"]["model"] = type2string(boundary_model)
    system_data["boundary_model"]["extrapolate_reference_values"] = boundary_model.extrapolate_reference_values
end

function add_system_data!(system_data, contact_model::HertzContactModel)
    system_data["contact_model"] = Dict{String, Any}()
    system_data["contact_model"]["model"] = type2string(contact_model)
    system_data["contact_model"]["elastic_modulus"] = contact_model.elastic_modulus
    system_data["contact_model"]["poissons_ratio"] = contact_model.poissons_ratio
end

function add_system_data!(system_data, contact_model::LinearContactModel)
    system_data["contact_model"] = Dict{String, Any}()
    system_data["contact_model"]["model"] = type2string(contact_model)
    system_data["contact_model"]["normal_stiffness"] = contact_model.normal_stiffness
end

function add_system_data!(system_data, state_equation::StateEquationCole)
    system_data["state_equation"] = Dict{String, Any}()
    system_data["state_equation"]["model"] = type2string(state_equation)
    system_data["state_equation"]["reference_density"] = state_equation.reference_density
    system_data["state_equation"]["background_pressure"] = state_equation.background_pressure
    system_data["state_equation"]["exponent"] = state_equation.exponent
end

function add_system_data!(system_data, state_equation::StateEquationIdealGas)
    system_data["state_equation"] = Dict{String, Any}()
    system_data["state_equation"]["model"] = type2string(state_equation)
    system_data["state_equation"]["reference_density"] = state_equation.reference_density
    system_data["state_equation"]["background_pressure"] = state_equation.background_pressure
    system_data["state_equation"]["gamma"] = state_equation.gamma
end

function add_system_data!(system_data, viscosity::Union{ViscosityAdami, ViscosityMorris})
    system_data["viscosity_model"] = Dict{String, Any}()
    system_data["viscosity_model"]["model"] = type2string(viscosity)
    system_data["viscosity_model"]["nu"] = viscosity.nu
    system_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_system_data!(system_data,
                          viscosity::Union{ViscosityAdamiSGS, ViscosityMorrisSGS})
    system_data["viscosity_model"] = Dict{String, Any}()
    system_data["viscosity_model"]["model"] = type2string(viscosity)
    system_data["viscosity_model"]["nu"] = viscosity.nu
    system_data["viscosity_model"]["C_S"] = viscosity.C_S
    system_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_system_data!(system_data, viscosity::ArtificialViscosityMonaghan)
    system_data["viscosity_model"] = Dict{String, Any}()
    system_data["viscosity_model"]["model"] = type2string(viscosity)
    system_data["viscosity_model"]["alpha"] = viscosity.alpha
    system_data["viscosity_model"]["beta"] = viscosity.beta
    system_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_system_data!(system_data,
                          density_diffusion::Union{DensityDiffusionAntuono,
                                                   DensityDiffusionMolteniColagrossi})
    system_data["density_diffusion"] = Dict{String, Any}()
    system_data["density_diffusion"]["model"] = type2string(density_diffusion)
    system_data["density_diffusion"]["delta"] = density_diffusion.delta
end

function add_system_data!(system_data, density_diffusion::DensityDiffusionFerrari)
    system_data["density_diffusion"] = Dict{String, Any}()
    system_data["density_diffusion"]["model"] = type2string(density_diffusion)
end

function add_system_data!(system_data, correction::AkinciFreeSurfaceCorrection)
    system_data["correction_method"] = Dict{String, Any}()
    system_data["correction_method"]["model"] = type2string(correction)
    system_data["correction_method"]["rho0"] = correction.rho0
end

function add_system_data!(system_data,
                          correction::Union{BlendedGradientCorrection, GradientCorrection,
                                            KernelCorrection, MixedKernelGradientCorrection,
                                            ShepardKernelCorrection})
    system_data["correction_method"] = Dict{String, Any}()
    system_data["correction_method"]["model"] = type2string(correction)
end

function add_system_data!(system_data,
                          surface_tension::Union{CohesionForceAkinci, SurfaceTensionAkinci,
                                                 SurfaceTensionMorris,
                                                 SurfaceTensionMomentumMorris})
    system_data["surface_tension"] = Dict{String, Any}()
    system_data["surface_tension"]["model"] = type2string(surface_tension)
    system_data["surface_tension"]["surface_tension_coefficient"] = surface_tension.surface_tension_coefficient
end

function add_system_data!(system_data, surface_normal_method::ColorfieldSurfaceNormal)
    system_data["surface_normal_method"] = Dict{String, Any}()
    system_data["surface_normal_method"]["model"] = type2string(surface_normal_method)
    system_data["surface_normal_method"]["boundary_contact_threshold"] = surface_normal_method.boundary_contact_threshold
    system_data["surface_normal_method"]["ideal_density_threshold"] = surface_normal_method.ideal_density_threshold
end

function add_system_data!(system_data, boundary_zone::BoundaryZone)
    system_data["boundary_zone"] = Dict{String, Any}()
    system_data["boundary_zone"]["boundary_type"] = type2string(boundary_zone.boundary_type)
    system_data["boundary_zone"]["zone_width"] = boundary_zone.zone_width
end

function add_system_data!(system_data, movement::BoundaryMovement)
    system_data["movement"] = Dict{String, Any}()
    system_data["movement"]["model"] = type2string(movement)
    system_data["movement"]["movement_function"] = type2string(movement.movement_function)
    system_data["movement"]["is_moving"] = type2string(movement.is_moving)
    system_data["movement"]["moving_particles"] = movement.moving_particles
end

function add_system_data!(system_data, penalty_force::PenaltyForceGanzenmueller)
    system_data["penalty_force"] = Dict{String, Any}()
    system_data["penalty_force"]["model"] = type2string(penalty_force)
    system_data["penalty_force"]["alpha"] = penalty_force.alpha
end

function add_system_data!(system_data, shifting_technique::TransportVelocityAdami)
    system_data["shifting_technique"] = Dict{String, Any}()
    system_data["shifting_technique"]["model"] = type2string(shifting_technique)
    system_data["shifting_technique"]["background_pressure"] = shifting_technique.background_pressure
end

function add_system_data!(system_data, shifting_technique::ParticleShiftingTechnique)
    system_data["shifting_technique"] = Dict{String, Any}()
    system_data["shifting_technique"]["model"] = type2string(shifting_technique)
end
