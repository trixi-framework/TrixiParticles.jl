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

    # fill `systems` with metadata for each system
    systems = Dict{String, Any}()
    foreach_system(semi) do system
        idx = system_indices(system, semi)
        name = add_opt_str_pre(prefix) * "$(names[idx])"

        meta_data = Dict{String, Any}()
        add_meta_data!(meta_data, system)

        systems[name] = meta_data
    end

    # initialize `simulation_meta_data` and add `systems`
    simulation_meta_data = Dict{String, Any}(
        "info" => Dict(
            "solver_name" => "TrixiParticles.jl",
            "solver_version" => git_hash,
            "julia_version" => string(VERSION)
        ),
        "systems" => systems
    )

    # write JSON-file
    output_directory = callback.output_directory
    mkpath(output_directory)

    json_file = joinpath(output_directory, "simulation_metadata.json")

    open(json_file, "w") do file
        JSON.print(file, simulation_meta_data, 2)
    end
end

function add_meta_data!(meta_data, system)
    return meta_data
end

function add_meta_data!(meta_data, system::DEMSystem)
    return meta_data
end

function add_meta_data!(meta_data, system::WeaklyCompressibleSPHSystem)
    # general `FLuidSystem`-Metadata
    meta_data["acceleration"] = system.acceleration
    meta_data["viscosity"] = type2string(system.viscosity)
    add_meta_data!(meta_data, system.viscosity)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length_factor"] = system.cache.smoothing_length_factor
    meta_data["density_calculator"] = type2string(system.density_calculator)

    # specific `WCSPH`-Metadata
    meta_data["solver"] = "WCSPH"
    meta_data["state_equation"] = type2string(system.state_equation)
    meta_data["state_equation_rho0"] = system.state_equation.reference_density
    meta_data["state_equation_pa"] = system.state_equation.background_pressure
    meta_data["state_equation_c"] = system.state_equation.sound_speed
    meta_data["correction_method"] = type2string(system.correction)
    if system.correction isa AkinciFreeSurfaceCorrection
        meta_data["correction_rho0"] = system.correction.rho0
    end
    if system.state_equation isa StateEquationCole
        meta_data["state_equation_exponent"] = system.state_equation.exponent
    end
    if system.state_equation isa StateEquationIdealGas
        meta_data["state_equation_gamma"] = system.state_equation.gamma
    end
end

function add_meta_data!(meta_data, system::EntropicallyDampedSPHSystem)
    # general `FLuidSystem`-Metadata
    meta_data["acceleration"] = system.acceleration
    meta_data["viscosity"] = type2string(system.viscosity)
    add_meta_data!(meta_data, system.viscosity)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length_factor"] = system.cache.smoothing_length_factor
    meta_data["density_calculator"] = type2string(system.density_calculator)

    # specific `EDAC`-Metadata
    meta_data["solver"] = "EDAC"
    meta_data["sound_speed"] = system.sound_speed
    meta_data["background_pressure_TVF"] = system.transport_velocity isa Nothing ? "-" :
                                           system.transport_velocity.background_pressure
end

add_meta_data!(meta_data, viscosity::Nothing) = meta_data

function add_meta_data!(meta_data, viscosity::Union{ViscosityAdami, ViscosityMorris})
    meta_data["viscosity_nu"] = viscosity.nu
    meta_data["viscosity_epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, viscosity::ArtificialViscosityMonaghan)
    meta_data["viscosity_alpha"] = viscosity.alpha
    meta_data["viscosity_beta"] = viscosity.beta
    meta_data["viscosity_epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, system::TotalLagrangianSPHSystem)
    meta_data["young_modulus"] = system.young_modulus
    meta_data["poisson_ratio"] = system.poisson_ratio
    meta_data["lame_lambda"] = system.lame_lambda
    meta_data["lame_mu"] = system.lame_mu
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length_factor"] = initial_smoothing_length(system) /
                                           particle_spacing(system, 1)

    add_meta_data!(meta_data, system.boundary_model, system)
end

function add_meta_data!(meta_data, system::OpenBoundarySPHSystem)
    meta_data["boundary_zone"] = type2string(system.boundary_zone)
    meta_data["width"] = round(system.boundary_zone.zone_width, digits=3)
    meta_data["velocity_function"] = type2string(system.reference_velocity)
    meta_data["pressure_function"] = type2string(system.reference_pressure)
    meta_data["density_function"] = type2string(system.reference_density)
end

function add_meta_data!(meta_data, system::BoundarySPHSystem)
    add_meta_data!(meta_data, system.boundary_model, system)
end

function add_meta_data!(meta_data, model::Nothing, system)
    return meta_data
end

function add_meta_data!(meta_data, model::BoundaryModelMonaghanKajtar, system)
    meta_data["boundary_model"] = "BoundaryModelMonaghanKajtar"
    meta_data["boundary_spacing_ratio"] = model.beta
    meta_data["boundary_K"] = model.K
end

function add_meta_data!(meta_data, model::BoundaryModelDummyParticles, system)
    meta_data["boundary_model"] = "BoundaryModelDummyParticles"
    meta_data["smoothing_kernel"] = type2string(model.smoothing_kernel)
    meta_data["smoothing_length"] = model.smoothing_length
    meta_data["density_calculator"] = type2string(model.density_calculator)
    meta_data["state_equation"] = type2string(model.state_equation)
    meta_data["viscosity_model"] = type2string(model.viscosity)
end

function add_meta_data!(meta_data, system::BoundaryDEMSystem)
    return meta_data
end
