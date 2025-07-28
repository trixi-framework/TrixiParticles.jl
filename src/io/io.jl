include("write_vtk.jl")
include("read_vtk.jl")

function write_meta_data(callback::Union{SolutionSavingCallback, PostprocessCallback},
                         integrator)
    # handle "_" on optional prefix strings
    add_opt_str_pre(str) = (str === "" ? "" : "$(str)_")

    git_hash = callback.git_hash
    prefix = hasproperty(callback, :prefix) ? callback.prefix : ""
    semi = integrator.p
    names = system_names(semi.systems)

    info = Dict{String, Any}()
    info["solver_name"] = "TrixiParticles.jl"
    info["solver_version"] = git_hash[]
    info["julia_version"] = string(VERSION)

    systems = Dict{String, Any}()
    foreach_system(semi) do system
        idx = system_indices(system, semi)
        name = add_opt_str_pre(prefix) * "$(names[idx])"

        meta_data = Dict{String, Any}()
        add_meta_data!(meta_data, system)

        systems[name] = meta_data
    end

    simulation_meta_data = Dict{String, Any}()
    simulation_meta_data["info"] = info
    simulation_meta_data["systems"] = systems

    # write JSON-file
    output_directory = callback.output_directory
    mkpath(output_directory)

    json_file = joinpath(output_directory, "simulation_meta_data.json")

    open(json_file, "w") do file
        JSON.print(file, simulation_meta_data, 2)
    end
end

add_meta_data!(meta_data, ::Nothing) = meta_data

function add_meta_data!(meta_data, data)
    ArgumentError("Unsupported data type: $(data). This data type is not implemented yet.")
end

function add_meta_data!(meta_data, system::FluidSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["particle_spacing"] = particle_spacing(system, 1)
    meta_data["density_calculator"] = type2string(system.density_calculator)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length"] = system.cache.smoothing_length
    meta_data["acceleration"] = system.acceleration
    meta_data["sound_speed"] = system_sound_speed(system)
    meta_data["pressure_acceleration_formulation"] = nameof(system.pressure_acceleration_formulation)
    add_meta_data!(meta_data, system.transport_velocity)
    add_meta_data!(meta_data, system.surface_tension)
    add_meta_data!(meta_data, system.surface_normal_method)
    add_meta_data!(meta_data, system.viscosity)
    add_meta_data!(meta_data, system.correction)
    add_meta_data!(meta_data, system_state_equation(system))
    if hasfield(typeof(system), :density_diffusion)
        add_meta_data!(meta_data, system.density_diffusion)
    end
    if hasfield(typeof(system), :alpha)
        meta_data["alpha"] = system.alpha
    end
end

function add_meta_data!(meta_data, system::TotalLagrangianSPHSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["particle_spacing"] = particle_spacing(system, 1)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length"] = system.smoothing_length
    meta_data["acceleration"] = system.acceleration
    add_meta_data!(meta_data, system.boundary_model)
    add_meta_data!(meta_data, system.penalty_force)
end

function add_meta_data!(meta_data, system::BoundarySPHSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["particle_spacing"] = particle_spacing(system, 1)
    meta_data["adhesion_coefficient"] = system.adhesion_coefficient
    add_meta_data!(meta_data, system.boundary_model)
    add_meta_data!(meta_data, system.movement)
end

function add_meta_data!(meta_data, system::BoundaryDEMSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["particle_spacing"] = particle_spacing(system, 1)
    meta_data["normal_stiffness"] = system.normal_stiffness
end

function add_meta_data!(meta_data, system::DEMSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["particle_spacing"] = particle_spacing(system, 1)
    meta_data["damping_coefficient"] = system.damping_coefficient
    meta_data["acceleration"] = system.acceleration
    add_meta_data!(meta_data, system.contact_model)
end

function add_meta_data!(meta_data, system::OpenBoundarySPHSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["particle_spacing"] = particle_spacing(system, 1)
    meta_data["reference_velocity"] = type2string(system.reference_velocity)
    meta_data["reference_pressure"] = type2string(system.reference_pressure)
    meta_data["reference_density"] = type2string(system.reference_density)
    add_meta_data!(meta_data, system.boundary_model)
    add_meta_data!(meta_data, system.boundary_zone)
end

function add_meta_data!(meta_data, boundary_model::BoundaryModelDummyParticles)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["model"] = type2string(boundary_model)
    meta_data["boundary_model"]["smoothing_kernel"] = type2string(boundary_model.smoothing_kernel)
    meta_data["boundary_model"]["smoothing_length"] = boundary_model.smoothing_length
    meta_data["boundary_model"]["density_calculator"] = type2string(boundary_model.density_calculator)
    add_meta_data!(meta_data["boundary_model"], boundary_model.state_equation)
    add_meta_data!(meta_data["boundary_model"], boundary_model.viscosity)
    add_meta_data!(meta_data["boundary_model"], boundary_model.correction)
end

function add_meta_data!(meta_data, boundary_model::BoundaryModelMonaghanKajtar)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["model"] = type2string(boundary_model)
    meta_data["boundary_model"]["beta"] = boundary_model.beta
    meta_data["boundary_model"]["K"] = boundary_model.K
    add_meta_data!(meta_data["boundary_model"], boundary_model.viscosity)
end

function add_meta_data!(meta_data, boundary_model::BoundaryModelTafuni)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["model"] = type2string(boundary_model)
end

function add_meta_data!(meta_data, boundary_model::BoundaryModelLastiwka)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["model"] = type2string(boundary_model)
    meta_data["boundary_model"]["extrapolate_reference_values"] = boundary_model.extrapolate_reference_values
end

function add_meta_data!(meta_data, contact_model::HertzContactModel)
    meta_data["contact_model"] = Dict{String, Any}()
    meta_data["contact_model"]["model"] = type2string(contact_model)
    meta_data["contact_model"]["elastic_modulus"] = contact_model.elastic_modulus
    meta_data["contact_model"]["poissons_ratio"] = contact_model.poissons_ratio
end

function add_meta_data!(meta_data, contact_model::LinearContactModel)
    meta_data["contact_model"] = Dict{String, Any}()
    meta_data["contact_model"]["model"] = type2string(contact_model)
    meta_data["contact_model"]["normal_stiffness"] = contact_model.normal_stiffness
end

function add_meta_data!(meta_data, state_equation::StateEquationCole)
    meta_data["state_equation"] = Dict{String, Any}()
    meta_data["state_equation"]["model"] = type2string(state_equation)
    meta_data["state_equation"]["reference_density"] = state_equation.reference_density
    meta_data["state_equation"]["background_pressure"] = state_equation.background_pressure
    meta_data["state_equation"]["exponent"] = state_equation.exponent
end

function add_meta_data!(meta_data, state_equation::StateEquationIdealGas)
    meta_data["state_equation"] = Dict{String, Any}()
    meta_data["state_equation"]["model"] = type2string(state_equation)
    meta_data["state_equation"]["reference_density"] = state_equation.reference_density
    meta_data["state_equation"]["background_pressure"] = state_equation.background_pressure
    meta_data["state_equation"]["gamma"] = state_equation.gamma
end

function add_meta_data!(meta_data, viscosity::Union{ViscosityAdami, ViscosityMorris})
    meta_data["viscosity_model"] = Dict{String, Any}()
    meta_data["viscosity_model"]["model"] = type2string(viscosity)
    meta_data["viscosity_model"]["nu"] = viscosity.nu
    meta_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, viscosity::Union{ViscosityAdamiSGS, ViscosityMorrisSGS})
    meta_data["viscosity_model"] = Dict{String, Any}()
    meta_data["viscosity_model"]["model"] = type2string(viscosity)
    meta_data["viscosity_model"]["nu"] = viscosity.nu
    meta_data["viscosity_model"]["C_S"] = viscosity.C_S
    meta_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, viscosity::ArtificialViscosityMonaghan)
    meta_data["viscosity_model"] = Dict{String, Any}()
    meta_data["viscosity_model"]["model"] = type2string(viscosity)
    meta_data["viscosity_model"]["alpha"] = viscosity.alpha
    meta_data["viscosity_model"]["beta"] = viscosity.beta
    meta_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data,
                        density_diffusion::Union{DensityDiffusionAntuono,
                                                 DensityDiffusionMolteniColagrossi})
    meta_data["density_diffusion"] = Dict{String, Any}()
    meta_data["density_diffusion"]["model"] = type2string(density_diffusion)
    meta_data["density_diffusion"]["delta"] = density_diffusion.delta
end

function add_meta_data!(meta_data, density_diffusion::DensityDiffusionFerrari)
    meta_data["density_diffusion"] = Dict{String, Any}()
    meta_data["density_diffusion"]["model"] = type2string(density_diffusion)
end

function add_meta_data!(meta_data, correction::AkinciFreeSurfaceCorrection)
    meta_data["correction_method"] = Dict{String, Any}()
    meta_data["correction_method"]["model"] = type2string(correction)
    meta_data["correction_method"]["rho0"] = correction.rho0
end

function add_meta_data!(meta_data,
                        correction::Union{BlendedGradientCorrection, GradientCorrection,
                                          KernelCorrection, MixedKernelGradientCorrection,
                                          ShepardKernelCorrection})
    meta_data["correction_method"] = Dict{String, Any}()
    meta_data["correction_method"]["model"] = type2string(correction)
end

function add_meta_data!(meta_data,
                        surface_tension::Union{CohesionForceAkinci, SurfaceTensionAkinci,
                                               SurfaceTensionMorris,
                                               SurfaceTensionMomentumMorris})
    meta_data["surface_tension"] = Dict{String, Any}()
    meta_data["surface_tension"]["model"] = type2string(surface_tension)
    meta_data["surface_tension"]["surface_tension_coefficient"] = surface_tension.surface_tension_coefficient
end

function add_meta_data!(meta_data, surface_normal_method::ColorfieldSurfaceNormal)
    meta_data["surface_normal_method"] = Dict{String, Any}()
    meta_data["surface_normal_method"]["model"] = type2string(surface_normal_method)
    meta_data["surface_normal_method"]["boundary_contact_threshold"] = surface_normal_method.boundary_contact_threshold
    meta_data["surface_normal_method"]["ideal_density_threshold"] = surface_normal_method.ideal_density_threshold
end

function add_meta_data!(meta_data, boundary_zone::BoundaryZone)
    meta_data["boundary_zone"] = Dict{String, Any}()
    meta_data["boundary_zone"]["boundary_type"] = type2string(boundary_zone.boundary_type)
    meta_data["boundary_zone"]["zone_width"] = boundary_zone.zone_width
end

function add_meta_data!(meta_data, movement::BoundaryMovement)
    meta_data["movement"] = Dict{String, Any}()
    meta_data["movement"]["model"] = type2string(movement)
    meta_data["movement"]["movement_function"] = type2string(movement.movement_function)
    meta_data["movement"]["is_moving"] = type2string(movement.is_moving)
    meta_data["movement"]["moving_particles"] = movement.moving_particles
end

function add_meta_data!(meta_data, penalty_force::PenaltyForceGanzenmueller)
    meta_data["penalty_force"] = Dict{String, Any}()
    meta_data["penalty_force"]["model"] = type2string(penalty_force)
    meta_data["penalty_force"]["alpha"] = penalty_force.alpha
end

function add_meta_data!(meta_data, transport_velocity::TransportVelocityAdami)
    meta_data["transport_velocity"] = Dict{String, Any}()
    meta_data["transport_velocity"]["model"] = type2string(transport_velocity)
    meta_data["transport_velocity"]["background_pressure"] = transport_velocity.background_pressure
end
