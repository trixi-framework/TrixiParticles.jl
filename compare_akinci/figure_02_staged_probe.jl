include("simulate.jl")

function transition_initial_condition(solution; system_index=1)
    semi = solution.prob.p.semi
    system = semi.systems[system_index]
    state = last(solution.u)
    v_ode, u_ode = state.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, last(solution.t))
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    particles = collect(eachparticle(system))

    coordinates = Array(TrixiParticles.current_coordinates(u, system))[:, particles]
    velocity = Array(TrixiParticles.current_velocity(v, system))[:, particles]
    density = [TrixiParticles.current_density(v, system, particle)
               for particle in particles]
    pressure = [TrixiParticles.current_pressure(v, system, particle)
                for particle in particles]
    mass = collect(system.mass[particles])
    particle_spacing = TrixiParticles.particle_spacing(system, first(particles))

    return InitialCondition(; coordinates, velocity, density, pressure, mass,
                            particle_spacing)
end

function segment_frames(solution)
    semi = solution.prob.p.semi
    return map(snapshot_frame, solution.u, Iterators.repeated(semi), solution.t)
end

function combined_solver_stats(solutions)
    stats = solver_stats.(solutions)
    return (; accepted_steps=sum(getproperty.(stats, :accepted_steps)),
            rejected_steps=sum(getproperty.(stats, :rejected_steps)))
end

function write_staged_figure_02_snapshot(config, output;
                                         particle_spacing=config.css.particle_spacing,
                                         shifting_stop_time=0.03)
    initial_time, final_time = config.kwargs.tspan
    initial_time < shifting_stop_time < config.kwargs.release_time ||
        throw(ArgumentError("shifting must stop between the initial and release times"))

    first_saveat = filter(time -> time <= shifting_stop_time,
                          config.kwargs.solution_saveat)
    first_kwargs = merge(config.kwargs,
                         (; tspan=(initial_time, shifting_stop_time),
                          solution_saveat=Tuple(first_saveat)))
    first_css = merge(config.css,
                      (; initial_particle_distribution=:lattice,
                       shifting_mode=:consistent_sun2019))
    first_config = merge(config,
                         (; name="$(config.name)_shifted_stage", kwargs=first_kwargs,
                          css=first_css))

    first_solution = nothing
    first_dt = nothing
    first_runtime = @elapsed begin
        first_solution, first_dt, _ = run_simulation(first_config; particle_spacing)
    end
    drop_initial_condition = transition_initial_condition(first_solution)

    second_saveat = filter(time -> time >= shifting_stop_time,
                           config.kwargs.solution_saveat)
    second_kwargs = merge(config.kwargs,
                          (; tspan=(shifting_stop_time, final_time),
                           solution_saveat=Tuple(second_saveat),
                           drop_initial_condition))
    second_css = merge(config.css,
                       (; initial_particle_distribution=:lattice,
                        shifting_mode=:none))
    second_config = merge(config,
                          (; name="$(config.name)_unshifted_stage", kwargs=second_kwargs,
                           css=second_css))

    second_solution = nothing
    second_dt = nothing
    second_runtime = @elapsed begin
        second_solution, second_dt, _ = run_simulation(second_config; particle_spacing)
    end

    first_frames = segment_frames(first_solution)
    second_frames = segment_frames(second_solution)
    duplicate_transition = isapprox(last(first_solution.t), first(second_solution.t))
    second_indices = duplicate_transition ? (2:length(second_frames)) :
                     eachindex(second_frames)
    frames = vcat(first_frames, second_frames[second_indices])
    times = vcat(collect(first_solution.t), collect(second_solution.t)[second_indices])
    accepted_dt = vcat(first_dt, second_dt)
    solutions = (first_solution, second_solution)
    snapshot = (; case_name="$(config.name)_staged",
                model=:css,
                surface_tension_coefficient=config.css.surface_tension_coefficient,
                particle_spacing,
                artificial_viscosity_alpha=config.css.artificial_viscosity_alpha,
                initial_particle_distribution=:lattice,
                packing_diagnostics=missing,
                shifting_mode=:staged_sun2019,
                shifting_stop_time,
                shifting_v_max_factor=config.css.shifting_v_max_factor,
                shifting_sound_speed_factor=config.css.shifting_sound_speed_factor,
                css_hydrodynamics=:modern,
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
        error("pass an output path, optional particle spacing, and optional shifting stop time")
    particle_spacing = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01 / 15
    shifting_stop_time = length(ARGS) == 3 ? parse(Float64, ARGS[3]) : 0.03
    write_staged_figure_02_snapshot(case_config("cube_to_sphere_css"), ARGS[1];
                                    particle_spacing, shifting_stop_time)
end
