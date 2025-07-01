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

function add_meta_data!(meta_data, system::FluidSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["density_calculator"] = type2string(system.density_calculator)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length_factor"] = system.cache.smoothing_length_factor
    meta_data["acceleration"] = system.acceleration
    meta_data["sound_speed"] = system_sound_speed(system)
    meta_data["background_pressure_TVF"] = system.transport_velocity isa Nothing ? "-" :
                                           system.transport_velocity.background_pressure
    add_meta_data!(meta_data, system.surface_tension)
    add_meta_data!(meta_data, system.viscosity)
    add_meta_data!(meta_data, system.correction)
    if system isa WeaklyCompressibleSPHSystem
        add_meta_data!(meta_data, system.state_equation)
    end
end

function add_meta_data!(meta_data, system::TotalLagrangianSPHSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["young_modulus"] = system.young_modulus
    meta_data["poisson_ratio"] = system.poisson_ratio
    meta_data["lame_lambda"] = system.lame_lambda
    meta_data["lame_mu"] = system.lame_mu
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length_factor"] = initial_smoothing_length(system) /
                                           particle_spacing(system, 1)
    meta_data["penalty_force"] = system.penalty_force
    meta_data["acceleration"] = system.acceleration
    add_meta_data!(meta_data, system.boundary_model)
end

function add_meta_data!(meta_data, system::OpenBoundarySPHSystem)
    meta_data["system_type"] = type2string(system)
    meta_data["reference_velocity"] = type2string(system.reference_velocity)
    meta_data["reference_pressure"] = type2string(system.reference_pressure)
    meta_data["reference_density"] = type2string(system.reference_density)
    add_meta_data!(meta_data, system.boundary_zone)
end

function add_meta_data!(meta_data, system::BoundarySPHSystem)
    meta_data["system_type"] = type2string(system)
    add_meta_data!(meta_data, system.boundary_model)
end

function add_meta_data!(meta_data, system::BoundaryDEMSystem)
    meta_data["system_type"] = type2string(system)
end

function add_meta_data!(meta_data, system::DEMSystem)
    meta_data["system_type"] = type2string(system)
end

function add_meta_data!(meta_data,
                        state_equation::Union{StateEquationCole, StateEquationIdealGas})
    meta_data["state_equation"] = Dict{String, Any}()
    meta_data["state_equation"]["model"] = type2string(state_equation)
    meta_data["state_equation"]["reference_density"] = state_equation.reference_density
    meta_data["state_equation"]["background_pressure"] = state_equation.background_pressure
    if state_equation isa StateEquationCole
        meta_data["state_equation"]["exponent"] = state_equation.exponent
    end
    if state_equation isa StateEquationIdealGas
        meta_data["state_equation"]["gamma"] = state_equation.gamma
    end
end

function add_meta_data!(meta_data, viscosity::Union{ViscosityAdami, ViscosityMorris})
    meta_data["viscosity_model"] = Dict{String, Any}()
    meta_data["viscosity_model"]["model"] = type2string(viscosity)
    meta_data["viscosity_model"]["nu"] = viscosity.nu
    meta_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, viscosity::ArtificialViscosityMonaghan)
    meta_data["viscosity_model"] = Dict{String, Any}()
    meta_data["viscosity_model"]["model"] = type2string(viscosity)
    meta_data["viscosity_model"]["alpha"] = viscosity.alpha
    meta_data["viscosity_model"]["beta"] = viscosity.beta
    meta_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, model::BoundaryModelMonaghanKajtar)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["model"] = "BoundaryModelMonaghanKajtar"
    meta_data["boundary_model"]["beta"] = model.beta
    meta_data["boundary_model"]["K"] = model.K
    add_meta_data!(meta_data["boundary_model"], model.viscosity)
end

function add_meta_data!(meta_data, model::BoundaryModelDummyParticles)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["model"] = "BoundaryModelDummyParticles"
    meta_data["boundary_model"]["smoothing_kernel"] = type2string(model.smoothing_kernel)
    meta_data["boundary_model"]["smoothing_length"] = model.smoothing_length
    meta_data["boundary_model"]["density_calculator"] = type2string(model.density_calculator)
    add_meta_data!(meta_data["boundary_model"], model.state_equation)
    add_meta_data!(meta_data["boundary_model"], model.viscosity)
    add_meta_data!(meta_data["boundary_model"], model.correction)
end

function add_meta_data!(meta_data,
                        correction::Union{AkinciFreeSurfaceCorrection,
                                          BlendedGradientCorrection,
                                          GradientCorrection, KernelCorrection,
                                          ShepardKernelCorrection})
    meta_data["correction_method"] = Dict{String, Any}()
    meta_data["correction_method"]["model"] = type2string(correction)
    if correction isa AkinciFreeSurfaceCorrection
        meta_data["correction_method"]["rho0"] = correction.rho0
    end
end

function add_meta_data!(meta_data,
                        surface_tension::Union{CohesionForceAkinci, SurfaceTensionAkinci,
                                               SurfaceTensionMorris,
                                               SurfaceTensionMomentumMorris})
    meta_data["surface_tension"] = Dict{String, Any}()
    meta_data["surface_tension"]["model"] = type2string(surface_tension)
    meta_data["surface_tension"]["surface_tension_coefficient"] = surface_tension.surface_tension_coefficient
end

function add_meta_data!(meta_data, boundary_zone)
    meta_data["boundary_zone"] = Dict{String, Any}()
    meta_data["boundary_zone"]["boundary_type"] = type2string(boundary_zone.boundary_type)
    meta_data["boundary_zone"]["zone_width"] = boundary_zone.zone_width
end
