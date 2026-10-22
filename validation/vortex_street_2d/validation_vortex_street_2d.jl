using TrixiParticles

tspan = (0.0, 20.0)

# In Tafuni et al. (2018), the resolution factor is `0.01` (5M particles), which result
# in 1.3M particles. `resolution_factor = 0.02` results in 200k particles and much noisier
# results compared to Tafuni et al. (2018).
resolution_factor = 0.05

reynolds_number = 200
cylinder_diameter = 0.1
domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)

mirror_method = FirstOrderMirroring(; firstorder_tolerance=1e-3)
open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method)

# The vortex shedding is chaotic, so the results depend on the order of the floating point
# operations. Use `SerialUpdate()` to obtain consistent results with multiple threads.
update_strategy = ParallelUpdate()

# Set this to a GPU backend like `CUDABackend()` to run the simulation on a GPU.
parallelization_backend = PolyesterBackend()

# Import variables into scope without running the simulation.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              reynolds_number=reynolds_number, saving_callback=nothing,
              open_boundary_model=open_boundary_model, update_strategy=update_strategy,
              particle_spacing_factor=resolution_factor,
              domain_size=domain_size, tspan=tspan, sol=nothing)

# The force on the cylinder is computed from the pairwise interaction forces between fluid
# and cylinder particles. With `tensile_instability_control`, the pressure force between
# fluid and cylinder vanishes where the pressure is negative, so the suction on the cylinder
# is not captured. Use the default momentum-conserving pressure formulation instead.
# The transport velocity formulation already prevents the tensile instability.
shifting_technique = TransportVelocityAdami(background_pressure=5 * fluid_density *
                                                                sound_speed^2)

# ==========================================================================================
# ==== Postprocessing
# To measure the hydrodynamic force on the cylinder with a `ThrustCalculator`, model the
# cylinder as a `TotalLagrangianSPHSystem` with only clamped particles.
# The fluid sees the same boundary model as with a `WallBoundarySystem`.
# The elastic parameters are irrelevant because no particles are integrated.
clamped_particles = eachparticle(cylinder)
boundary_system_cylinder = TotalLagrangianSPHSystem(cylinder; smoothing_kernel,
                                                    smoothing_length, clamped_particles,
                                                    young_modulus=1e4, poisson_ratio=0.3,
                                                    boundary_model=boundary_model_cylinder)

# Set up the simulation with the TLSPH cylinder without running it, so that the force
# calculators can be created for the cylinder system in the final semidiscretization.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              parallelization_backend=parallelization_backend,
              reynolds_number=reynolds_number,
              open_boundary_model=open_boundary_model, update_strategy=update_strategy,
              shifting_technique=shifting_technique,
              boundary_system_cylinder=boundary_system_cylinder,
              particle_spacing_factor=resolution_factor, domain_size=domain_size,
              tspan=tspan, saving_callback=nothing, sol=nothing)

# The `Semidiscretization` creates a copy of the TLSPH system, so the `ThrustCalculator`
# needs the system as it is stored in the semidiscretization.
cylinder_system = semi.systems[end]

# The force coefficients are computed from the total hydrodynamic force on the cylinder,
# including pressure and viscous forces.
let cylinder_system = cylinder_system, semi = semi
    drag_calculator = ThrustCalculator(cylinder_system, semi,
                                       direction=(1.0, 0.0))
    lift_calculator = ThrustCalculator(cylinder_system, semi,
                                       direction=(0.0, 1.0))
    force_scaling = 2 / (fluid_density * prescribed_velocity^2 * cylinder_diameter)

    global function drag_coefficient(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        force = drag_calculator(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        isnothing(force) && return nothing

        return force_scaling * force
    end

    global function lift_coefficient(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        force = lift_calculator(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        isnothing(force) && return nothing

        return force_scaling * force
    end
end

# The force coefficients are very noisy at low resolutions. A velocity sensor in the wake
# of the cylinder yields a much cleaner signal of the vortex shedding, so we additionally
# record the transverse velocity `v_y` at a single point on the wake centerline.
# On the centerline, `v_y` oscillates with the shedding frequency itself
# (whereas `v_x` oscillates with twice the shedding frequency).
velocity_sensor_point = [cylinder_center[1] + 2 * cylinder_diameter; cylinder_center[2];;]

wake_velocity_y(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

let velocity_sensor_point = velocity_sensor_point
    global function wake_velocity_y(system::TrixiParticles.AbstractFluidSystem,
                                    dv_ode, du_ode, v_ode, u_ode, semi, t)
        values = interpolate_points(velocity_sensor_point, semi, system, v_ode, u_ode;
                                    cut_off_bnd=false)

        return values.velocity[2, 1]
    end
end

# Tag the output file with the resolution, e.g. `resulting_force_dp20.json` for a particle
# spacing of `d/20`.
resolution_name = round(Int, 1 / resolution_factor)

pp_callback = PostprocessCallback(; dt=0.02,
                                  lift_coefficient, drag_coefficient, wake_velocity_y,
                                  filename="resulting_force_dp$(resolution_name)",
                                  write_csv=false, write_file_interval=10)

# ======================================================================================
# ==== Run the simulation
callbacks = CallbackSet(callbacks, pp_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # May need tuning to prevent boundary penetration
            reltol=1e-4, # May need tuning to prevent boundary penetration
            save_everystep=false, callback=callbacks, maxiters=10^7);
