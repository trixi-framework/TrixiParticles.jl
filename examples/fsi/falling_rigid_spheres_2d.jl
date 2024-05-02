using TrixiParticles
using OrdinaryDiffEq
using LinearAlgebra

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.01
solid_particle_spacing = fluid_particle_spacing

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 4.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 0.5)
tank_size = (2.0, 2.0)

fluid_density = 1000.0
sound_speed = 150
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

sphere1_radius = 0.3
sphere2_radius = 0.2
# wood
sphere1_density = 600.0
# steel
#sphere2_density = 7700
sphere2_density = 3000

sphere1_center = (0.5, 1.1)
sphere2_center = (1.0, 0.8)
sphere1 = SphereShape(solid_particle_spacing, sphere1_radius, sphere1_center,
                      sphere1_density, sphere_type=VoxelSphere())
sphere2 = SphereShape(solid_particle_spacing, sphere2_radius, sphere2_center,
                      sphere2_density, sphere_type=VoxelSphere())

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 3.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ViscosityAdami(nu=1e-4)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length,
                                             viscosity=viscosity)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Solid

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites_1 = fluid_density * ones(size(sphere1.density))
hydrodynamic_masses_1 = hydrodynamic_densites_1 * solid_particle_spacing^ndims(fluid_system)

solid_boundary_model_1 = BoundaryModelDummyParticles(hydrodynamic_densites_1,
                                                     hydrodynamic_masses_1,
                                                     state_equation=state_equation,
                                                     boundary_density_calculator,
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     viscosity=viscosity)

hydrodynamic_densites_2 = fluid_density * ones(size(sphere2.density))
hydrodynamic_masses_2 = hydrodynamic_densites_2 * solid_particle_spacing^ndims(fluid_system)

solid_boundary_model_2 = BoundaryModelDummyParticles(hydrodynamic_densites_2,
                                                     hydrodynamic_masses_2,
                                                     state_equation=state_equation,
                                                     boundary_density_calculator,
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     viscosity=viscosity)

solid_system_1 = RigidSPHSystem(sphere1; boundary_model=solid_boundary_model_1,
                                acceleration=(0.0, -gravity),
                                particle_spacing=fluid_particle_spacing)
solid_system_2 = RigidSPHSystem(sphere2; boundary_model=solid_boundary_model_2,
                                acceleration=(0.0, -gravity),
                                particle_spacing=fluid_particle_spacing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(boundary_system, solid_system_2, fluid_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out", prefix="",
                                         write_meta_data=true)

function collision(vu, integrator, semi, t)
    v_ode, u_ode = vu.x

    # println("vollision at ", t)

    TrixiParticles.collision_interaction!(v_ode, u_ode, semi)

    # TrixiParticles.foreach_system(semi) do system
    #     v =  TrixiParticles.wrap_v(v_ode, system, semi)
    #     u =  TrixiParticles.wrap_u(u_ode, system, semi)

    #     if system isa RigidSPHSystem && system.has_collided.value
    #         velocity_change = norm(v[:, 1]) -  norm(v[:, 1] + system.collision_impulse)
    #         if abs(velocity_change) > 1
    #             println("before: ", v[:, 1])
    #             println("after: ", v[:, 1] + system.collision_impulse)
    #             println("imp: ", system.collision_impulse)

    #             exit(-1)
    #         end
    #         for particle in  TrixiParticles.each_moving_particle(system)
    #             v[:, particle] += system.collision_impulse
    #             u[:, particle] += system.collision_u
    #         end
    #     end
    # end
end

# Use a Runge-Kutta method with automatic (error based) time step size control.
# sol = solve(ode, RDPK3SpFSAL49(;stage_limiter! =collision),
#             abstol=1e-7, # Default abstol is 1e-6
#             reltol=1e-5, # Default reltol is 1e-3
#             save_everystep=false, callback=callbacks);
stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition=false; (step_limiter!)=collision),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);
