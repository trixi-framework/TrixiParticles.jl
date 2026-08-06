include("figure_02_staged_probe.jl")

function write_surface_switch_snapshot(output; particle_spacing=0.01 / 15,
                                       switch_time=nothing)
    base = case_config("cube_to_sphere_css")
    switch_time = isnothing(switch_time) ? base.kwargs.release_time : switch_time
    initial_time, final_time = base.kwargs.tspan
    initial_time < switch_time < final_time ||
        throw(ArgumentError("switch time must lie inside the simulation interval"))

    common_css = (; pressure_stabilization=:interface_tic,
                  tic_strength=0.25,
                  shifting_mode=:consistent_sun2019,
                  contact_angle=nothing,
                  initial_particle_distribution=:lattice)

    first_saveat = filter(time -> time <= switch_time, base.kwargs.solution_saveat)
    first_kwargs = merge(base.kwargs,
                         (; tspan=(initial_time, switch_time),
                          solution_saveat=Tuple(first_saveat)))
    first_css = merge(base.css, common_css,
                      (; surface_tension_mode=:c_csf,
                       ccsf_contact_angle=nothing,
                       normal_smoothing=false))
    first_config = merge(base,
                         (; name="cube_to_sphere_c_csf_stage",
                          kwargs=first_kwargs, css=first_css))

    first_solution = nothing
    first_dt = nothing
    first_runtime = @elapsed begin
        first_solution, first_dt, _ = run_simulation(first_config; particle_spacing)
    end

    drop_initial_condition = transition_initial_condition(first_solution)
    second_saveat = filter(time -> time >= switch_time, base.kwargs.solution_saveat)
    second_kwargs = merge(base.kwargs,
                          (; tspan=(switch_time, final_time),
                           solution_saveat=Tuple(second_saveat),
                           drop_initial_condition))
    second_css = merge(base.css, common_css,
                       (; surface_tension_mode=:css,
                        ccsf_contact_angle=nothing,
                        normal_smoothing=true))
    second_config = merge(base,
                          (; name="cube_to_sphere_smoothed_css_stage",
                           kwargs=second_kwargs, css=second_css))

    second_solution = nothing
    second_dt = nothing
    second_runtime = @elapsed begin
        second_solution, second_dt, _ = run_simulation(second_config; particle_spacing)
    end

    first_frames = segment_frames(first_solution)
    second_frames = segment_frames(second_solution)
    duplicate_switch = isapprox(last(first_solution.t), first(second_solution.t))
    second_indices = duplicate_switch ? (2:length(second_frames)) : eachindex(second_frames)
    frames = vcat(first_frames, second_frames[second_indices])
    times = vcat(collect(first_solution.t), collect(second_solution.t)[second_indices])
    accepted_dt = vcat(first_dt, second_dt)
    solutions = (first_solution, second_solution)

    snapshot = (; case_name="cube_to_sphere_c_csf_to_smoothed_css",
                model=:css,
                surface_tension_coefficient=base.css.surface_tension_coefficient,
                particle_spacing,
                artificial_viscosity_alpha=base.css.artificial_viscosity_alpha,
                surface_tension_mode=:c_csf_to_smoothed_css,
                smoothing_kernel_mode=base.css.smoothing_kernel_mode,
                smoothing_length_ratio=base.css.smoothing_length_ratio,
                normal_smoothing=:post_switch,
                contact_angle=nothing,
                ccsf_contact_angle=nothing,
                viscosity_mode=base.css.viscosity_mode,
                density_diffusion_mode=base.css.density_diffusion_mode,
                density_diffusion_delta=base.css.density_diffusion_delta,
                initial_particle_distribution=:lattice,
                packing_diagnostics=missing,
                pressure_stabilization=:interface_tic,
                tic_strength=0.25,
                shifting_mode=:consistent_sun2019,
                shifting_v_max_factor=base.css.shifting_v_max_factor,
                shifting_sound_speed_factor=base.css.shifting_sound_speed_factor,
                css_hydrodynamics=:modern,
                switch_time,
                runtime=first_runtime + second_runtime,
                solver_stats=combined_solver_stats(solutions),
                timestep_stats=timestep_stats(accepted_dt),
                times, frames)

    open(output, "w") do io
        serialize(io, snapshot)
    end
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in 1:3 ||
        error("pass an output path, optional particle spacing, and optional switch time")
    particle_spacing = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01 / 15
    switch_time = length(ARGS) == 3 ? parse(Float64, ARGS[3]) : nothing
    write_surface_switch_snapshot(ARGS[1]; particle_spacing, switch_time)
end
