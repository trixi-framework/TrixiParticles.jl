using TrixiParticles

tspan = (0.0, 20.0)

# In Tafuni et al. (2018), the resolution factor is `0.01` (d/dx = 100, 5M particles).
# `resolution_factor = 0.02` results in 1.3M particles. The default `0.05` results in
# 200k particles and much noisier results compared to Tafuni et al. (2018).
resolution_factor = 0.05

reynolds_number = 200
cylinder_diameter = 0.1

# Use the original domain size and boundary model from Tafuni et al. (2018)
# instead of the smaller domain and the default boundary model of the example.
domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)
mirror_method = FirstOrderMirroring(; firstorder_tolerance=1e-3)
open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method)

# The vortex shedding is chaotic, so the results depend on the order of the floating point
# operations. Use `SerialUpdate()` to obtain consistent results with multiple threads.
update_strategy = ParallelUpdate()

# Set this to a GPU backend like `CUDABackend()` to run the simulation on a GPU.
parallelization_backend = PolyesterBackend()

# Unlike Tafuni et al. (2018), we prescribe the pressure at the outlet in addition to the
# inflow velocity, as in the example. Tafuni et al. only prescribe the inflow velocity and
# extrapolate the outlet pressure from the fluid. Then the pressure level in the domain is
# not fixed by the boundary conditions and can drift. In our experiments, this setup was
# fragile. It only remained stable at high resolutions. At lower resolutions, the pressure
# drifted with all shifting techniques we tried, and most simulations crashed.
outlet_reference_pressure = 0.0

# Overrides that are shared by both `trixi_include` calls of the example below.
example_overrides = (; reynolds_number, open_boundary_model, outlet_reference_pressure,
                     update_strategy, particle_spacing_factor=resolution_factor,
                     domain_size, tspan, saving_callback=nothing, sol=nothing)

# Import variables into scope without running the simulation.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl");
              example_overrides..., ode=nothing, semi=nothing)

# Replace the consistent particle shifting technique of the example by the original
# Particle Shifting Technique by Sun et al. (2017). Like the shifting used by
# Tafuni et al. (2018), this is a position correction that does not conserve volume
# and adds no additional terms to the momentum equation, which could affect the measured
# forces. Shifting also prevents the tensile instability without
# `tensile_instability_control`, which would corrupt the measured forces.
shifting_technique = ParticleShiftingTechniqueSun2017()

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
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl");
              example_overrides..., parallelization_backend, shifting_technique,
              boundary_system_cylinder)

# The `Semidiscretization` creates a copy of the TLSPH system, so the `ThrustCalculator`
# needs the system as it is stored in the semidiscretization.
# The TLSPH system is the last system in the semidiscretization.
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

# Only record the wake velocity for the fluid system.
wake_velocity_y(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

let velocity_sensor_point = velocity_sensor_point
    global function wake_velocity_y(system::TrixiParticles.AbstractFluidSystem,
                                    dv_ode, du_ode, v_ode, u_ode, semi, t)
        values = interpolate_points(velocity_sensor_point, semi, system, v_ode, u_ode;
                                    cut_off_bnd=false)

        return values.velocity[2, 1]
    end
end

# Monitor the volume loss caused by the non-conservative shifting (see above).
# Since all fluid particles have the same mass `m`, the particle count `N` is proportional
# to the total mass in the domain. The total volume of the fluid particles can be computed
# in postprocessing as `N * m / average_density`. This is exact up to the (negligible)
# difference between the harmonic and the arithmetic mean of the density.
function n_fluid_particles(system::TrixiParticles.AbstractFluidSystem,
                           dv_ode, du_ode, v_ode, u_ode, semi, t)
    return length(TrixiParticles.each_active_particle(system))
end

# Only record this quantity for the fluid system.
n_fluid_particles(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

function average_density(system::TrixiParticles.AbstractFluidSystem,
                         dv_ode, du_ode, v_ode, u_ode, semi, t)
    return TrixiParticles.avg_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
end

# Only record this quantity for the fluid system.
average_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

# Tag the output file with the resolution, e.g. `validation_result_vortex_street_20.json`
# for a particle spacing of `d/20`.
particles_per_diameter = round(Int, 1 / resolution_factor)
output_filename = "validation_result_vortex_street_$(particles_per_diameter)"

pp_callback = PostprocessCallback(; dt=0.02,
                                  lift_coefficient, drag_coefficient, wake_velocity_y,
                                  n_fluid_particles, average_density,
                                  filename=output_filename,
                                  write_csv=false, write_file_interval=10)

# ==========================================================================================
# ==== Run the simulation
callbacks = CallbackSet(callbacks, pp_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # May need tuning to prevent boundary penetration
            reltol=1e-4, # May need tuning to prevent boundary penetration
            save_everystep=false, callback=callbacks, maxiters=10^7);
