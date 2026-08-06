using OrdinaryDiffEqLowStorageRK
using Serialization
using Statistics
using TrixiParticles

include("cases.jl")
include("figure_02_packing.jl")
isdefined(@__MODULE__, :FreeSurfaceDensityDiffusionAntuono) ||
    include("wcsph_variants.jl")

function css_smoothing_kernel(mode)
    mode == :wendland_c2 && return WendlandC2Kernel{3}()
    mode == :wendland_c4 && return WendlandC4Kernel{3}()
    mode == :wendland_c6 && return WendlandC6Kernel{3}()
    throw(ArgumentError("unknown CSS smoothing kernel mode '$mode'"))
end

function simulation_kwargs(config; surface_tension_coefficient=nothing,
                           particle_spacing=nothing, css_hydrodynamics=:modern)
    if !hasproperty(config, :css)
        isnothing(surface_tension_coefficient) ||
            throw(ArgumentError("a CSS coefficient override requires a CSS case"))
        return isnothing(particle_spacing) ? config.kwargs :
               merge(config.kwargs, (; particle_spacing))
    end

    css = config.css
    coefficient = something(surface_tension_coefficient,
                            css.surface_tension_coefficient)
    spacing = something(particle_spacing, css.particle_spacing)
    coefficient > 0 ||
        throw(ArgumentError("the CSS surface-tension coefficient must be positive"))
    spacing > 0 || throw(ArgumentError("the particle spacing must be positive"))
    css.smoothing_length_ratio > 0 ||
        throw(ArgumentError("the CSS smoothing-length ratio must be positive"))
    smoothing_kernel = css_smoothing_kernel(css.smoothing_kernel_mode)
    smoothing_length = css.smoothing_length_ratio * spacing
    contact_model = isnothing(css.contact_angle) ? nothing :
                    WettedAreaContactAngle(css.contact_angle)
    boundary_contact_threshold = isnothing(contact_model) ? Inf : 0.1
    surface_tension,
    surface_normal_method = if css.surface_tension_mode == :css
        (SurfaceTensionMomentumMorris(; surface_tension_coefficient=coefficient),
         ColorfieldSurfaceNormal(; boundary_contact_threshold,
                                 interface_threshold=0.01,
                                 ideal_density_threshold=0.95,
                                 normal_smoothing=css.normal_smoothing,
                                 contact_model))
    elseif css.surface_tension_mode == :c_csf
        isnothing(contact_model) ||
            throw(ArgumentError("C-CSF boundary contact is not implemented"))
        (SurfaceTensionMorris(; surface_tension_coefficient=coefficient),
         CorrectedCSFSurfaceNormal(; contact_angle=css.ccsf_contact_angle))
    else
        throw(ArgumentError("unknown CSS surface tension mode '$(css.surface_tension_mode)'"))
    end

    if css_hydrodynamics == :akinci_baseline
        isnothing(contact_model) ||
            throw(ArgumentError("Akinci-baseline CSS hydrodynamics do not support contact"))
        css.surface_tension_mode == :css ||
            throw(ArgumentError("Akinci-baseline hydrodynamics require CSS surface tension"))
        return merge(config.kwargs,
                     (; particle_spacing=spacing,
                      surface_tension,
                      surface_normal_method))
    elseif css_hydrodynamics != :modern
        throw(ArgumentError("unknown CSS hydrodynamics '$css_hydrodynamics'"))
    end

    shifting_technique = if css.shifting_mode == :none
        nothing
    elseif css.shifting_mode == :sun2017
        ParticleShiftingTechniqueSun2017(;
                                         free_surface_treatment=FreeSurfaceTangentialShifting())
    elseif css.shifting_mode == :consistent_sun2019
        ConsistentShiftingSun2019(;
                                  v_max_factor=css.shifting_v_max_factor,
                                  sound_speed_factor=css.shifting_sound_speed_factor,
                                  free_surface_treatment=FreeSurfaceTangentialShifting())
    else
        throw(ArgumentError("unknown CSS shifting mode '$(css.shifting_mode)'"))
    end
    pressure_acceleration,
    fluid_clip_negative_pressure = if css.pressure_stabilization == :none
        (nothing, true)
    elseif css.pressure_stabilization == :interface_tic
        (InterfaceAwareTensileInstabilityControl(; strength=css.tic_strength), false)
    else
        throw(ArgumentError("unknown CSS pressure stabilization '$(css.pressure_stabilization)'"))
    end
    update_callback = css.shifting_mode == :sun2017 ? UpdateCallback() : nothing
    density_diffusion = if css.density_diffusion_mode == :none
        nothing
    elseif css.density_diffusion_mode == :antuono
        DensityDiffusionAntuono(; delta=css.density_diffusion_delta)
    elseif css.density_diffusion_mode == :free_surface_antuono
        FreeSurfaceDensityDiffusionAntuono(; delta=css.density_diffusion_delta,
                                           reference_density=1000.0)
    else
        throw(ArgumentError("unknown CSS density diffusion mode '$(css.density_diffusion_mode)'"))
    end
    equivalent_nu = css.artificial_viscosity_alpha * smoothing_length * 40.0 / 10
    viscosity = if css.viscosity_mode == :artificial_monaghan
        ArtificialViscosityMonaghan(; alpha=css.artificial_viscosity_alpha, beta=0.0)
    elseif css.viscosity_mode == :morris
        ViscosityMorris(; nu=equivalent_nu)
    elseif css.viscosity_mode == :adami
        ViscosityAdami(; nu=equivalent_nu)
    else
        throw(ArgumentError("unknown CSS viscosity mode '$(css.viscosity_mode)'"))
    end

    return merge(config.kwargs,
                 (; particle_spacing=spacing,
                  smoothing_kernel,
                  smoothing_length,
                  provide_boundary_surface_geometry=!isnothing(contact_model) ||
                                                    !isnothing(css.ccsf_contact_angle),
                  density_calculator=ContinuityDensity(),
                  density_diffusion,
                  correction=nothing,
                  pressure_acceleration,
                  fluid_clip_negative_pressure,
                  update_callback,
                  viscosity,
                  shifting_technique,
                  surface_tension,
                  surface_normal_method))
end

function run_simulation(config; surface_tension_coefficient=nothing,
                        particle_spacing=nothing, css_hydrodynamics=:modern)
    module_name = Symbol("AkinciComparison_", config.name)
    simulation_module = Module(module_name)
    Core.eval(simulation_module, :(using TrixiParticles))
    example = joinpath(examples_dir(), "fluid", config.example)
    kwargs = simulation_kwargs(config; surface_tension_coefficient, particle_spacing,
                               css_hydrodynamics)
    packing_diagnostics = missing
    if config.example == "akinci_cube_to_sphere_3d.jl" && hasproperty(config, :css)
        distribution = config.css.initial_particle_distribution
        if distribution == :packed
            drop_initial_condition,
            packing_diagnostics = packed_cube_initial_condition(;
                                                                particle_spacing=kwargs.particle_spacing,
                                                                relative_jitter=config.css.packing_relative_jitter,
                                                                seed=config.css.packing_seed,
                                                                maxiters=config.css.packing_maxiters)
            kwargs = merge(kwargs, (; drop_initial_condition))
        elseif distribution == :jittered
            cube_side_length = 0.01
            n_cube = ntuple(_ -> round(Int, cube_side_length / kwargs.particle_spacing), 3)
            cube_min = (-cube_side_length / 2, -cube_side_length / 2, 0.0025)
            lattice = RectangularShape(kwargs.particle_spacing, n_cube, cube_min;
                                       density=1000.0)
            drop_initial_condition = jittered_initial_condition(lattice;
                                                                relative_amplitude=config.css.packing_relative_jitter,
                                                                seed=config.css.packing_seed)
            packing_diagnostics = (; relative_jitter=config.css.packing_relative_jitter,
                                   seed=config.css.packing_seed, maxiters=0)
            kwargs = merge(kwargs, (; drop_initial_condition))
        elseif distribution != :lattice
            throw(ArgumentError("unknown initial particle distribution '$distribution'"))
        end
    end
    accepted_dt = Float64[]
    if config.example == "akinci_cube_to_sphere_3d.jl"
        timestep_diagnostic_callback = DiscreteCallback((_, time, _) -> time > 0,
                                                        integrator -> begin
                                                            push!(accepted_dt,
                                                                  abs(integrator.t -
                                                                      integrator.tprev))
                                                            u_modified!(integrator, false)
                                                        end;
                                                        save_positions=(false, false))
        kwargs = merge(kwargs, (; timestep_diagnostic_callback))
    end
    trixi_include(simulation_module, example; kwargs...)
    solution = Base.invokelatest(Core.eval, simulation_module, :sol)
    return solution, accepted_dt, packing_diagnostics
end

function snapshot_frame(state, semi, time)
    v_ode, u_ode = state.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, time)

    systems = map(eachindex(semi.systems)) do system_index
        system = semi.systems[system_index]
        particles = collect(eachparticle(system))
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coordinates = Array(TrixiParticles.active_coordinates(u, system))
        spacing = isempty(particles) ? nothing :
                  TrixiParticles.particle_spacing(system, first(particles))
        kind = if system isa TrixiParticles.AbstractFluidSystem
            :fluid
        elseif system isa TrixiParticles.AbstractBoundarySystem
            :boundary
        elseif system isa TrixiParticles.AbstractStructureSystem
            :structure
        else
            :other
        end
        pressure = kind == :fluid ?
                   [TrixiParticles.current_pressure(v, system, particle)
                    for particle in particles] : nothing
        density = kind == :fluid ?
                  [TrixiParticles.current_density(v, system, particle)
                   for particle in particles] : nothing
        velocity = kind == :fluid ?
                   Array(TrixiParticles.current_velocity(v, system))[:, particles] : nothing
        return (; coordinates, particle_spacing=spacing, kind, pressure, density, velocity)
    end

    return (; systems)
end

function solver_stats(solution)
    stats = hasproperty(solution, :stats) ? solution.stats : solution.destats
    accepted_steps = hasproperty(stats, :naccept) ? stats.naccept : missing
    rejected_steps = hasproperty(stats, :nreject) ? stats.nreject : missing
    return (; accepted_steps, rejected_steps)
end

function timestep_stats(accepted_dt)
    length(accepted_dt) > 6 ||
        return (; samples=length(accepted_dt), p01=missing, median=missing,
                tail_to_head=missing)
    samples = accepted_dt[6:(end - 1)]
    window = max(1, floor(Int, 0.2 * length(samples)))
    return (; samples=length(samples), p01=quantile(samples, 0.01),
            median=median(samples),
            tail_to_head=median(samples[(end - window + 1):end]) /
                         median(samples[1:window]))
end

function write_snapshot(config, output; surface_tension_coefficient=nothing,
                        particle_spacing=nothing, css_hydrodynamics=:modern)
    result = nothing
    runtime = @elapsed result = run_simulation(config; surface_tension_coefficient,
                                               particle_spacing, css_hydrodynamics)
    solution, accepted_dt, packing_diagnostics = result
    semi = solution.prob.p.semi
    frames = map(snapshot_frame, solution.u, Iterators.repeated(semi), solution.t)
    model = hasproperty(config, :css) ? :css : :akinci
    coefficient = hasproperty(config, :css) ?
                  something(surface_tension_coefficient,
                            config.css.surface_tension_coefficient) : missing
    configured_spacing = hasproperty(config, :css) ?
                         something(particle_spacing, config.css.particle_spacing) : missing
    artificial_viscosity_alpha = hasproperty(config, :css) ?
                                 config.css.artificial_viscosity_alpha : missing
    surface_tension_mode = hasproperty(config, :css) ?
                           config.css.surface_tension_mode : missing
    smoothing_kernel_mode = hasproperty(config, :css) ?
                            config.css.smoothing_kernel_mode : missing
    smoothing_length_ratio = hasproperty(config, :css) ?
                             config.css.smoothing_length_ratio : missing
    normal_smoothing = hasproperty(config, :css) ? config.css.normal_smoothing : missing
    contact_angle = hasproperty(config, :css) ? config.css.contact_angle : missing
    ccsf_contact_angle = hasproperty(config, :css) ? config.css.ccsf_contact_angle : missing
    viscosity_mode = hasproperty(config, :css) ? config.css.viscosity_mode : missing
    density_diffusion_mode = hasproperty(config, :css) ?
                             config.css.density_diffusion_mode : missing
    density_diffusion_delta = hasproperty(config, :css) ?
                              config.css.density_diffusion_delta : missing
    initial_particle_distribution = hasproperty(config, :css) ?
                                    config.css.initial_particle_distribution : missing
    pressure_stabilization = hasproperty(config, :css) &&
                             css_hydrodynamics == :modern ?
                             config.css.pressure_stabilization : missing
    tic_strength = hasproperty(config, :css) &&
                   css_hydrodynamics == :modern ?
                   config.css.tic_strength : missing
    shifting_mode = hasproperty(config, :css) &&
                    css_hydrodynamics == :modern ?
                    config.css.shifting_mode : missing
    shifting_v_max_factor = hasproperty(config, :css) &&
                            css_hydrodynamics == :modern ?
                            config.css.shifting_v_max_factor : missing
    shifting_sound_speed_factor = hasproperty(config, :css) &&
                                  css_hydrodynamics == :modern ?
                                  config.css.shifting_sound_speed_factor : missing
    snapshot = (; case_name=config.name, model,
                surface_tension_coefficient=coefficient,
                particle_spacing=configured_spacing,
                artificial_viscosity_alpha,
                surface_tension_mode,
                smoothing_kernel_mode,
                smoothing_length_ratio,
                normal_smoothing,
                contact_angle,
                ccsf_contact_angle,
                viscosity_mode,
                density_diffusion_mode,
                density_diffusion_delta,
                initial_particle_distribution,
                packing_diagnostics,
                pressure_stabilization,
                tic_strength,
                shifting_mode,
                shifting_v_max_factor,
                shifting_sound_speed_factor,
                css_hydrodynamics=hasproperty(config, :css) ? css_hydrodynamics : missing,
                runtime, solver_stats=solver_stats(solution),
                timestep_stats=timestep_stats(accepted_dt),
                times=collect(solution.t), frames)

    open(output, "w") do io
        serialize(io, snapshot)
    end
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (2, 3, 4) ||
        error("pass a comparison case, output path, optional CSS coefficient, " *
              "and optional particle spacing")
    coefficient = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : nothing
    particle_spacing = length(ARGS) == 4 ? parse(Float64, ARGS[4]) : nothing
    write_snapshot(case_config(ARGS[1]), ARGS[2];
                   surface_tension_coefficient=coefficient, particle_spacing)
end
