using TrixiParticles

# ==========================================================================================
# ==== Resolution
# P. Ramachandran, K. Puri
# "Entropically damped artiﬁcial compressibility for SPH".
# In: Computers and Fluids, Volume 179 (2019), pages 579-594.
# https://doi.org/10.1016/j.compfluid.2018.11.023
# The paper provides reference data for particle spacings particle_spacings = [0.02, 0.01, 0.005]
particle_spacing = 0.02

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 5.0)
reynolds_number = 100.0

density_calculators = [ContinuityDensity(), SummationDensity()]
perturb_coordinates = [false, true]

function compute_l1v_error(system, v_ode, u_ode, semi, t)
    v_analytical_avg = 0.0
    v_avg = 0.0

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    for particle in TrixiParticles.eachparticle(system)
        position = TrixiParticles.current_coords(u, system, particle)

        v_mag = TrixiParticles.norm(TrixiParticles.current_velocity(v, system, particle))
        v_analytical = TrixiParticles.norm(velocity_function(position, t))

        v_analytical_avg += abs(v_analytical)
        v_avg += abs(v_mag - v_analytical)
    end
    v_analytical_avg /= nparticles(system)

    v_avg /= nparticles(system)

    return v_avg /= v_analytical_avg
end

function compute_l1p_error(system, v_ode, u_ode, semi, t)
    p_max_exact = 0.0

    L1p = 0.0

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    for particle in TrixiParticles.eachparticle(system)
        position = TrixiParticles.current_coords(u, system, particle)

        # compute pressure error
        p_analytical = pressure_function(position, t)
        p_max_exact = max(p_max_exact, abs(p_analytical))

        # p_computed - p_average
        p_computed = TrixiParticles.current_pressure(v, system, particle) -
                     TrixiParticles.average_pressure(system, particle)
        L1p += abs(p_computed - p_analytical)
    end

    L1p /= nparticles(system)

    return L1p /= p_max_exact
end

# The pressure plotted in the paper is the difference of the local pressure minus
# the average of the pressure of all particles.
function diff_p_loc_p_avg(system, v, u, semi, t)
    p_avg_tot = avg_pressure(system, v, u, semi, t)

    return v[end, :] .- p_avg_tot
end

for density_calculator in density_calculators, perturbation in perturb_coordinates,
    wcsph in [false, true]
    n_particles_xy = round(Int, 1.0 / particle_spacing)

    name_density_calculator = density_calculator isa SummationDensity ?
                              "summation_density" : "continuity_density"
    name_perturbation = perturbation ? "_perturbed" : ""

    output_directory = joinpath("out_tgv", name_density_calculator * name_perturbation,
                                wcsph ? "wcsph" : "edac",
                                "validation_run_taylor_green_vortex_2d_nparticles_$(n_particles_xy)x$(n_particles_xy)")
    saving_callback = SolutionSavingCallback(dt=0.02,
                                             output_directory=output_directory,
                                             p_avg=diff_p_loc_p_avg)

    pp_callback = PostprocessCallback(; dt=0.02,
                                      L1v=compute_l1v_error,
                                      L1p=compute_l1p_error,
                                      output_directory=output_directory,
                                      filename="errors",
                                      write_csv=true, write_file_interval=1)

    # Import variables into scope
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "taylor_green_vortex_2d.jl"),
                  density_calculator=density_calculator,
                  perturb_coordinates=perturbation, wcsph=wcsph,
                  particle_spacing=particle_spacing, reynolds_number=reynolds_number,
                  tspan=tspan, saving_callback=saving_callback, pp_callback=pp_callback)
end
