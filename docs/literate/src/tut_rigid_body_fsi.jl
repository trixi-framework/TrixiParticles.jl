# # [Fluid-structure interaction with rigid body SPH](@id tut_rigid_body_fsi)

# This tutorial introduces fluid-structure interaction (FSI) with [`RigidBodySystem`](@ref).
# We build a simplified version of
# [`examples/fsi/falling_rotating_rigid_squares_2d.jl`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/examples/fsi/falling_rotating_rigid_squares_2d.jl),
# where two rigid squares with initial angular velocity fall into a water tank.
# Compared to the example file, we use a coarser resolution to keep the documentation build fast.

using TrixiParticles
using OrdinaryDiffEq

# ## Resolution

# As in the other tutorials, we start by defining the particle spacing.
# We use the same spacing for the fluid and the rigid bodies so that the coupling
# operates on the same particle resolution.
fluid_particle_spacing = 0.04
structure_particle_spacing = fluid_particle_spacing

# We keep three boundary layers for the tank walls.
boundary_layers = 3
spacing_ratio = 1
nothing # hide

# ## Experiment setup

# The setup is a rectangular tank with an open top.
gravity = 9.81
tspan = (0.0, 0.6)

initial_fluid_size = (2.0, 1.0)
tank_size = (2.0, 2.0)

fluid_density = 1000.0
sound_speed = 100.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)
nothing # hide

# Next, we define two rigid squares.
# Their physical density determines whether they tend to float or sink,
# while the initial angular velocity prescribes the rigid-body rotation.
square1_side_length = 0.4
square2_side_length = 0.3

square1_density = 600.0
square2_density = 2000.0

square1_nparticles_side = round(Int, square1_side_length / structure_particle_spacing)
square2_nparticles_side = round(Int, square2_side_length / structure_particle_spacing)

square1_bottom_left = (0.4, 1.25)
square2_bottom_left = (1.25, 1.30)

square1_angular_velocity = 5.0
square2_angular_velocity = -7.5

square1 = RectangularShape(structure_particle_spacing,
                           (square1_nparticles_side, square1_nparticles_side),
                           square1_bottom_left,
                           density=square1_density)
square2 = RectangularShape(structure_particle_spacing,
                           (square2_nparticles_side, square2_nparticles_side),
                           square2_bottom_left,
                           density=square2_density)

square1 = apply_angular_velocity(square1, square1_angular_velocity)
square2 = apply_angular_velocity(square2, square2_angular_velocity)
nothing # hide

# We can visualize the initial setup before defining the SPH systems.
using Plots
plot(tank.fluid, tank.boundary, square1, square2,
     labels=["fluid" "boundary" "square 1" "square 2"])
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_setup.png"); # hide
nothing # hide
# ![initial setup](tut_rigid_body_fsi_setup.png)

# ## Fluid system

# For the water, we use a standard WCSPH discretization.
fluid_smoothing_length = 1.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity))
nothing # hide

# ## Boundary system

# The tank walls are represented by boundary particles that interact with the fluid.
boundary_density_calculator = AdamiPressureExtrapolation()
tank_boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                  tank.boundary.mass,
                                                  state_equation=state_equation,
                                                  boundary_density_calculator,
                                                  fluid_smoothing_kernel,
                                                  fluid_smoothing_length)

boundary_system = WallBoundarySystem(tank.boundary, tank_boundary_model)
nothing # hide

# ## Rigid body systems

# For rigid-body FSI, the rigid particles need two separate pieces of information:
#
# 1. their physical density, which is already stored in `square1` and `square2`
#    and determines the rigid-body mass and inertia,
# 2. a boundary model for the fluid coupling, which requires hydrodynamic
#    masses and densities on the rigid-body surface.
#
# In this tutorial, we initialize the hydrodynamic density from the surrounding fluid density.
function rigid_body_boundary_model(shape)
    hydrodynamic_densities = fluid_density * ones(size(shape.density))
    hydrodynamic_masses = hydrodynamic_densities *
                          structure_particle_spacing^ndims(fluid_system)

    return BoundaryModelDummyParticles(hydrodynamic_densities,
                                       hydrodynamic_masses,
                                       state_equation=state_equation,
                                       boundary_density_calculator,
                                       fluid_smoothing_kernel,
                                       fluid_smoothing_length)
end

square1_boundary_model = rigid_body_boundary_model(square1)
square2_boundary_model = rigid_body_boundary_model(square2)
nothing # hide

# The `boundary_model` that we pass to `RigidBodySystem` is therefore not the rigid-body material law.
# It is only the interface seen by the fluid.
# The rigid-body mass, center of mass, and moment of inertia still come from the original
# particle masses and densities stored in `square1` and `square2`.
#
# Besides fluid coupling, rigid bodies can also use a dedicated contact model for
# rigid-wall and rigid-rigid collisions.
# This contact model is independent of the SPH boundary model above.
#
# In `RigidContactModel`, the arguments have the following meaning:
#
# - `normal_stiffness`: stiffness of the normal penalty spring.
#   Larger values reduce visible overlap during contact, but also make the contact problem stiffer,
#   which can require smaller time steps.
# - `normal_damping`: damping in the normal direction.
#   This removes kinetic energy during impact.
#   Larger values make collisions less bouncy, while `0.0` gives the most elastic response.
# - `contact_distance`: activation distance of the contact shell around each rigid particle.
#   Contact forces start to act once particles or walls come closer than this distance.
#   If it is left at `0.0`, TrixiParticles.jl falls back to the `particle_spacing`
#   passed to `RigidBodySystem`.
#
# Here we choose a moderate damping and a contact distance of two particle spacings
# so that contact is detected robustly on this coarse tutorial resolution.
contact_model = RigidContactModel(; normal_stiffness=2.0e5,
                                  normal_damping=150.0,
                                  contact_distance=2.0 *
                                                   structure_particle_spacing)

# The first argument of [`RigidBodySystem`](@ref) is the sampled rigid-body particle cloud.
# It provides the initial coordinates, particle masses, material densities, and any initial velocity.
# In our case, the initial angular velocity was already written into the initial condition
# via [`apply_angular_velocity`](@ref).
#
# The main keyword arguments used here are:
#
# - `boundary_model`: the fluid-facing boundary representation of the rigid body.
#   Without it, the rigid body would not exchange pressure and viscous forces with the fluid.
# - `contact_model`: enables rigid-wall and rigid-rigid collision forces.
#   If this is omitted, the body still participates in FSI, but rigid contact is disabled.
# - `acceleration`: constant body force per unit mass, here gravity.
# - `particle_spacing`: reference spacing used internally for copying the contact model
#   and for contact-related time-step estimates.
#   This should usually match the spacing used to sample the rigid shape.
#
# [`RigidBodySystem`](@ref) also offers more advanced keywords such as `max_manifolds`,
# `source_terms`, `adhesion_coefficient`, and `color_value`.
# We keep them at their defaults here because the present example only needs
# basic rigid-body dynamics, FSI coupling, and contact handling.
rigid_body_system_1 = RigidBodySystem(square1;
                                      boundary_model=square1_boundary_model,
                                      contact_model=contact_model,
                                      acceleration=(0.0, -gravity),
                                      particle_spacing=structure_particle_spacing)
rigid_body_system_2 = RigidBodySystem(square2;
                                      boundary_model=square2_boundary_model,
                                      contact_model=contact_model,
                                      acceleration=(0.0, -gravity),
                                      particle_spacing=structure_particle_spacing)
nothing # hide

# ## Semidiscretization

# The semidiscretization couples the fluid, the tank wall boundary, and both rigid bodies.
semi = Semidiscretization(fluid_system, boundary_system,
                          rigid_body_system_1, rigid_body_system_2)
ode = semidiscretize(semi, tspan)
nothing # hide

# ## Time integration

# We reuse the standard callbacks from the other tutorials.
info_callback = InfoCallback(interval=20)
saving_callback = SolutionSavingCallback(dt=0.05)

callbacks = CallbackSet(info_callback, saving_callback)
nothing # hide

# For this FSI problem, a small enough `dtmax` is useful during the free-fall phase,
# so that fluid particles do not tunnel through the rigid bodies or the tank wall on impact.
# ```@cast @__NAME__; width=100, height=50, delay=0, loop=true, loop_delay=5
# sol = solve(ode, RDPK3SpFSAL49(),
#             abstol=1e-6,
#             reltol=1e-4,
#             dtmax=2e-3,
#             save_everystep=false, callback=callbacks);
# ```
sol = solve(ode, RDPK3SpFSAL49(), #!md
            abstol=1e-6, #!md
            reltol=1e-4, #!md
            dtmax=2e-3, #!md
            save_everystep=false, callback=callbacks) #!md

# We can inspect the final state with Plots.jl.
plot(sol)
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_plot.png"); # hide
nothing # hide
# ![solution plot](tut_rigid_body_fsi_plot.png)

# The lighter square tends to stay closer to the free surface, while the denser square sinks.
# This same setup pattern extends directly to other shapes and more rigid bodies.
#
# ## Next steps
#
# A natural extension is to add a layer of small rigid spheres on top of the initial water column.
# If we choose a density of `500.0`, the spheres are lighter than the surrounding fluid and should
# mostly float while interacting with the waves and the falling squares.
# The example file [`examples/fsi/falling_rotating_rigid_squares_2d.jl`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/examples/fsi/falling_rotating_rigid_squares_2d.jl)
# contains the same idea as a commented-out block.
#
# One possible implementation is:
# ```julia
# small_sphere_radius = 0.08
# small_sphere_density = 500.0
# small_sphere_y = initial_fluid_size[2] + small_sphere_radius
# small_sphere_x_positions = 0.25:(3 * small_sphere_radius):1.75
#
# small_spheres = [SphereShape(structure_particle_spacing, small_sphere_radius,
#                              (x, small_sphere_y), small_sphere_density,
#                              sphere_type=RoundSphere())
#                  for x in small_sphere_x_positions]
#
# small_sphere_systems = [begin
#     sphere_boundary_model = rigid_body_boundary_model(sphere)
#     RigidBodySystem(sphere;
#                     boundary_model=sphere_boundary_model,
#                     contact_model=contact_model,
#                     acceleration=(0.0, -gravity),
#                     particle_spacing=structure_particle_spacing)
# end for sphere in small_spheres]
#
# semi = Semidiscretization(fluid_system, boundary_system,
#                           rigid_body_system_1, rigid_body_system_2,
#                           small_sphere_systems...)
# ```
#
# This modification is useful for studying many-body rigid-body FSI, including
# floating-particle rafts, repeated rigid-rigid contact, and wave-driven rearrangement.
