"""
	trixi2json(solution_callback, integrator)

Write simulation metadata to a JSON-file.

# Arguments
- `solution_callback`:  Callback storing metadata and output settings.
- `integrator`:         ODE integrator containing simulation data.

# Example
```jldoctest; output = false, saving_callback = SolutionSavingCallback(dt = 0.1, output_directory = "output", prefix = "solution"),
setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan = (0.0, 0.01), callbacks = saving_callback))

# output

```
"""

function trixi2json(solution_callback, integrator)
    semi = integrator.p
    (; systems) = semi

    output_directory = solution_callback.output_directory
    prefix = solution_callback.prefix
    git_hash = solution_callback.git_hash

    filenames = system_names(systems)

    foreach_system(semi) do system
        system_index = system_indices(system, semi)

        trixi2json(system; system_name=filenames[system_index], output_directory, prefix,
                   git_hash)
    end
end

function trixi2json(system; system_name, output_directory, prefix, git_hash)
    mkpath(output_directory)

    meta_data = Dict{String, Any}(
        "solver_name" => "TrixiParticles.jl",
        "solver_version" => git_hash,
        "julia_version" => string(VERSION)
    )

    get_meta_data!(meta_data, system)

    # handle "_" on optional prefix strings
    add_opt_str_pre(str) = (str === "" ? "" : "$(str)_")

    # Write metadata to JSON-file
    json_file = joinpath(output_directory,
                         add_opt_str_pre(prefix) * "$(system_name)_metadata.json")

    open(json_file, "w") do file
        JSON.print(file, meta_data, 2)
    end
end

function get_meta_data!(meta_data, system::FluidSystem)
    meta_data["acceleration"] = system.acceleration
    meta_data["viscosity"] = type2string(system.viscosity)
    get_meta_data!(meta_data, system.viscosity)
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length"] = system.smoothing_length
    meta_data["density_calculator"] = type2string(system.density_calculator)

    if system isa WeaklyCompressibleSPHSystem
        meta_data["state_equation"] = type2string(system.state_equation)
        meta_data["state_equation_rho0"] = system.state_equation.reference_density
        meta_data["state_equation_pa"] = system.state_equation.background_pressure
        meta_data["state_equation_c"] = system.state_equation.sound_speed
        meta_data["solver"] = "WCSPH"

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
    else
        meta_data["solver"] = "EDAC"
        meta_data["sound_speed"] = system.sound_speed
        meta_data["background_pressure_TVF"] = system.transport_velocity isa Nothing ? "-" :
                                               system.transport_velocity.background_pressure
    end

    return meta_data
end

get_meta_data!(meta_data, viscosity::Nothing) = meta_data

function get_meta_data!(meta_data, viscosity::Union{ViscosityAdami, ViscosityMorris})
    meta_data["viscosity_nu"] = viscosity.nu
    meta_data["viscosity_epsilon"] = viscosity.epsilon
end

function get_meta_data!(meta_data, viscosity::ArtificialViscosityMonaghan)
    meta_data["viscosity_alpha"] = viscosity.alpha
    meta_data["viscosity_beta"] = viscosity.beta
    meta_data["viscosity_epsilon"] = viscosity.epsilon
end

function get_meta_data!(meta_data, system::TotalLagrangianSPHSystem)
    meta_data["young_modulus"] = system.young_modulus
    meta_data["poisson_ratio"] = system.poisson_ratio
    meta_data["lame_lambda"] = system.lame_lambda
    meta_data["lame_mu"] = system.lame_mu
    meta_data["smoothing_kernel"] = type2string(system.smoothing_kernel)
    meta_data["smoothing_length"] = system.smoothing_length

    get_meta_data!(meta_data, system.boundary_model, system)
end

function get_meta_data!(meta_data, system::OpenBoundarySPHSystem)
    meta_data["boundary_zone"] = type2string(system.boundary_zone)
    meta_data["width"] = round(system.boundary_zone.zone_width, digits=3)
    meta_data["flow_direction"] = system.flow_direction
    meta_data["velocity_function"] = type2string(system.reference_velocity)
    meta_data["pressure_function"] = type2string(system.reference_pressure)
    meta_data["density_function"] = type2string(system.reference_density)
end

function get_meta_data!(meta_data, system::BoundarySPHSystem)
    get_meta_data!(meta_data, system.boundary_model, system)
end

function get_meta_data!(meta_data, model, system)
    return meta_data
end

function get_meta_data!(meta_data, model::BoundaryModelMonaghanKajtar, system)
    meta_data["boundary_model"] = "BoundaryModelMonaghanKajtar"
    meta_data["boundary_spacing_ratio"] = model.beta
    meta_data["boundary_K"] = model.K
end

function get_meta_data!(meta_data, model::BoundaryModelDummyParticles, system)
    meta_data["boundary_model"] = "BoundaryModelDummyParticles"
    meta_data["smoothing_kernel"] = type2string(model.smoothing_kernel)
    meta_data["smoothing_length"] = model.smoothing_length
    meta_data["density_calculator"] = type2string(model.density_calculator)
    meta_data["state_equation"] = type2string(model.state_equation)
    meta_data["viscosity_model"] = type2string(model.viscosity)
end

function get_meta_data!(meta_data, system::BoundaryDEMSystem)
    return meta_data
end
