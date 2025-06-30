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

function add_meta_data!(meta_data, system)
    return meta_data
end

function add_meta_data!(meta_data, system::WeaklyCompressibleSPHSystem)
    # general `FluidSystem`-metadata
    meta_data["phase"] = "fluid"
    meta_data["type"] = type2string(system)
    meta_data["acceleration"] = system.acceleration
    add_meta_data!(meta_data, system.viscosity)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length_factor"] = system.cache.smoothing_length_factor
    meta_data["density_calculator"] = type2string(system.density_calculator)

    # specific `WCSPH`-metadata
    meta_data["state_equation"] = Dict{String, Any}()
    meta_data["state_equation"]["type"] = type2string(system.state_equation)
    meta_data["state_equation"]["rho0"] = system.state_equation.reference_density
    meta_data["state_equation"]["pa"] = system.state_equation.background_pressure
    meta_data["state_equation"]["c"] = system.state_equation.sound_speed
    if system.state_equation isa StateEquationCole
        meta_data["state_equation"]["exponent"] = system.state_equation.exponent
    end
    if system.state_equation isa StateEquationIdealGas
        meta_data["state_equation"]["gamma"] = system.state_equation.gamma
    end

    meta_data["correction_method"] = Dict{String, Any}()
    meta_data["correction_method"]["type"] = type2string(system.correction)
    if system.correction isa AkinciFreeSurfaceCorrection
        meta_data["correction_method"]["rho0"] = system.correction.rho0
    end
end

function add_meta_data!(meta_data, system::EntropicallyDampedSPHSystem)
    # general `FluidSystem`-metadata
    meta_data["phase"] = "fluid"
    meta_data["type"] = type2string(system)
    meta_data["acceleration"] = system.acceleration
    add_meta_data!(meta_data, system.viscosity)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length_factor"] = system.cache.smoothing_length_factor
    meta_data["density_calculator"] = type2string(system.density_calculator)

    # specific `EDAC`-metadata
    meta_data["sound_speed"] = system.sound_speed
    meta_data["background_pressure_TVF"] = system.transport_velocity isa Nothing ? "-" :
                                           system.transport_velocity.background_pressure
end

add_meta_data!(meta_data, viscosity::Nothing) = meta_data

function add_meta_data!(meta_data, viscosity::Union{ViscosityAdami, ViscosityMorris})
    meta_data["viscosity_model"] = Dict{String, Any}()
    meta_data["viscosity_model"]["type"] = type2string(viscosity)
    meta_data["viscosity_model"]["nu"] = viscosity.nu
    meta_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, viscosity::ArtificialViscosityMonaghan)
    meta_data["viscosity_model"] = Dict{String, Any}()
    meta_data["viscosity_model"]["type"] = type2string(viscosity)
    meta_data["viscosity_model"]["alpha"] = viscosity.alpha
    meta_data["viscosity_model"]["beta"] = viscosity.beta
    meta_data["viscosity_model"]["epsilon"] = viscosity.epsilon
end

function add_meta_data!(meta_data, system::TotalLagrangianSPHSystem)
    meta_data["phase"] = "boundary"
    meta_data["type"] = type2string(system)
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
    meta_data["phase"] = "boundary"
    meta_data["type"] = type2string(system)
    meta_data["type"] = type2string(system.boundary_zone.boundary_type)
    meta_data["width"] = round(system.boundary_zone.zone_width, digits=3)
    meta_data["velocity_function"] = type2string(system.reference_velocity)
    meta_data["pressure_function"] = type2string(system.reference_pressure)
    meta_data["density_function"] = type2string(system.reference_density)
end

function add_meta_data!(meta_data, system::BoundarySPHSystem)
    meta_data["type"] = type2string(system)
    add_meta_data!(meta_data, system.boundary_model, system)
end

function add_meta_data!(meta_data, model::Nothing, system)
    return meta_data
end

function add_meta_data!(meta_data, model::BoundaryModelMonaghanKajtar, system)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["type"] = "BoundaryModelMonaghanKajtar"
    meta_data["boundary_model"]["spacing_ratio"] = model.beta
    meta_data["boundary_model"]["K"] = model.K
end

function add_meta_data!(meta_data, model::BoundaryModelDummyParticles, system)
    meta_data["boundary_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["type"] = "BoundaryModelDummyParticles"
    meta_data["boundary_model"]["smoothing_kernel"] = type2string(model.smoothing_kernel)
    meta_data["boundary_model"]["smoothing_length"] = model.smoothing_length
    meta_data["boundary_model"]["density_calculator"] = type2string(model.density_calculator)

    meta_data["boundary_model"]["state_equation"] = Dict{String, Any}()
    meta_data["boundary_model"]["state_equation"]["type"] = type2string(model.state_equation)

    meta_data["boundary_model"]["viscosity_model"] = Dict{String, Any}()
    meta_data["boundary_model"]["viscosity_model"]["type"] = type2string(model.viscosity)
end

function add_meta_data!(meta_data, system::BoundaryDEMSystem)
    meta_data["type"] = type2string(system)
    return meta_data
end

function add_meta_data!(meta_data, system::DEMSystem)
    meta_data["type"] = type2string(system)
    return meta_data
end
