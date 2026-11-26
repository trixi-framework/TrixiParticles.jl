using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using CSV: CSV
using LinearAlgebra: dot

# Compare the same moving-particle problem for each correction configuration.
# Particles per axis => Adami viscosity regularization, fixed across corrections.
setups = (16 => 0.01, 32 => 0.01, 64 => 0.001)
reynolds_number = 100.0
t_end = 0.5
n_intervals = 20
timing_repeats = 3
output_directory = joinpath("out", "corrections_periodic", "tgv_2d")
cases = ((:none, nothing, nothing), (:shepard, ShepardKernelCorrection(), nothing),
         (:gradient, nothing, GradientCorrection()),
         (:shepard_gradient, ShepardKernelCorrection(), GradientCorrection()),
         (:mixed, nothing, MixedKernelGradientCorrection()),
         (:shepard_mixed, ShepardKernelCorrection(), MixedKernelGradientCorrection()))

function tgv_diagnostics(sol, analytical_velocity, nu, method, resolution)
    semi = sol.prob.p.semi
    system = only(semi.systems)
    energies = map(sol.u, sol.t) do state, t
        v_ode, u_ode = state.x
        return TrixiParticles.kinetic_energy(system, nothing, nothing, v_ode, u_ode, semi,
                                             t)
    end
    return map(eachindex(sol.t)) do i
        t = sol.t[i]
        v_ode, u_ode = sol.u[i].x
        velocity_l2_error = GC.@preserve v_ode u_ode begin
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            err2 = ref2 = 0.0
            for a in TrixiParticles.eachparticle(system)
                x = TrixiParticles.current_coords(u, system, a)
                exact = analytical_velocity(x, t)
                delta = TrixiParticles.current_velocity(v, system, a) - exact
                err2 += system.mass[a] * dot(delta, delta)
                ref2 += system.mass[a] * dot(exact, exact)
            end
            sqrt(err2 / ref2)
        end
        energy_ratio = energies[i] / first(energies)
        # On the unit square the velocity amplitude decays as exp(-8π²νt).
        energy_ratio_exact = exp(-16pi^2 * nu * t)
        return (; method, resolution, nu, epsilon=system.viscosity.epsilon, time=t,
                kinetic_energy=energies[i], energy_ratio, energy_ratio_exact,
                velocity_l2_error)
    end
end

function tgv_solve_sample(semi, tspan, options)
    # Reconstruct the same initial problem (including mutable caches) outside the timer.
    problem = semidiscretize(deepcopy(semi), tspan)
    opts = deepcopy(options)
    GC.gc()
    seconds = @elapsed solution = solve(problem, RDPK3SpFSAL49(); opts...)
    solution.retcode == ReturnCode.Success && solution.t ≈ opts.saveat ||
        error("Incomplete timed Taylor–Green run: $(solution.retcode)")
    return solution, seconds
end

timing_repeats >= 1 || throw(ArgumentError("use at least one timing repeat"))
sample_times = range(0.0, t_end; length=n_intervals + 1)
decay_results = NamedTuple[]
timing_results = NamedTuple[]
for (n, eps_visc) in setups, (method, dc, gc) in cases
    n >= 12 ||
        throw(ArgumentError("use at least 12 particles per axis for periodic support"))
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "taylor_green_vortex_2d.jl");
                  wcsph=false, density_calculator=SummationDensity(),
                  particle_spacing=1.0 / n, perturb_coordinates=false,
                  smoothing_kernel=WendlandC6Kernel{2}(), smoothing_length=2.0 / n,
                  density_correction=dc, gradient_correction=gc, reynolds_number,
                  viscosity=ViscosityAdami(nu=1.0 / reynolds_number, epsilon=eps_visc),
                  tspan=(0.0, t_end), saveat=collect(sample_times),
                  info_callback=nothing, saving_callback=nothing, ode=nothing, sol=nothing)
    # Use the untimed warm-up for accuracy; record only subsequent solves for performance.
    solution, _ = tgv_solve_sample(semi, tspan, solve_options)
    append!(decay_results,
            tgv_diagnostics(solution, velocity_function, nu, method, n))
    for trial in 1:timing_repeats
        timed_solution, seconds = tgv_solve_sample(semi, tspan, solve_options)
        push!(timing_results,
              (; method, resolution=n, epsilon=eps_visc, t_end, trial, seconds,
               accepted_steps=timed_solution.stats.naccept, threads=Threads.nthreads(),
               cpu=Sys.CPU_NAME, julia_version=string(VERSION)))
    end
end
mkpath(output_directory)
decay_csv = joinpath(output_directory, "tgv_decay.csv")
CSV.write(decay_csv, decay_results)
timing_csv = joinpath(output_directory, "tgv_timings.csv")
CSV.write(timing_csv, timing_results)
println("Wrote $decay_csv and $timing_csv")
