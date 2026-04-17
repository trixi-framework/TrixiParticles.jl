# # [Fluid-structure interaction with rigid bodies](@id tut_rigid_body_fsi)

# This tutorial introduces fluid-structure interaction (FSI) with the [`RigidBodySystem`](@ref).
# We build a simplified version of
# [`examples/fsi/falling_rotating_rigid_squares_2d.jl`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/examples/fsi/falling_rotating_rigid_squares_2d.jl),
# where two rigid squares with initial angular velocity fall into a water tank.
# Compared to the example file, we use a coarser resolution to see results quickly.
# For more details on the general setup of tank and fluid, see [the tutorial on setting up a simulation](@ref tut_setup).

# We will build up the simulation step by step:
# 1. Rigid bodies without any interaction.
# 2. Fluid-structure interaction, but without rigid-rigid or rigid-wall contact.
# 3. Full simulation with fluid-structure interaction and contact.
# 4. Using a different geometry.

using TrixiParticles
using OrdinaryDiffEq
using Plots

# ## Resolution and basic setup
# As in the other tutorials, we start by defining the particle spacing.
# We use the same spacing for the fluid and the rigid bodies so that the coupling
# operates on the same particle resolution.
fluid_particle_spacing = 0.04
structure_particle_spacing = fluid_particle_spacing
nothing #hide

boundary_layers = 3
spacing_ratio = 1
nothing # hide

# The setup is a rectangular tank with an open top.
gravity = 9.81
tspan = (0.0, 1.0)
tspan2 = (0.0, 0.6) # shorter simulation time required for some simulations below

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

# ## Rigid body geometry
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

# We can visualize the initial setup.
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

# ## Step 1: Without boundary model and contact model
# In the first step, we simulate the rigid bodies without interaction with the fluid or the tank.
# This means we will create a `Semidiscretization` containing only the rigid body systems.
# The bodies will move according to the prescribed gravity and initial velocities, but without
# any collisions or fluid forces.

rigid_body_system_1_step1 = RigidBodySystem(square1; acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
rigid_body_system_2_step1 = RigidBodySystem(square2; acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
nothing # hide

# Note that the `Semidiscretization` does not contain the fluid or boundary systems.
semi_step1 = Semidiscretization(rigid_body_system_1_step1, rigid_body_system_2_step1)
ode_step1 = semidiscretize(semi_step1, tspan2)
nothing # hide

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="step1")
callbacks = CallbackSet(info_callback, saving_callback)
nothing # hide

# ```julia
# sol_step1 = solve(ode_step1, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks)
# ```
sol_step1 = solve(ode_step1, RDPK3SpFSAL49(), save_everystep=false) # hide
nothing # hide

#
# Let's plot the final state of this simulation.
# We can see the final positions of the squares. The fluid and tank are plotted
# in the background to show that no interaction has taken place.
# As you can see, the squares fall through the fluid and the tank walls.
p = plot(tank.fluid, tank.boundary, labels=["fluid" "boundary"])
plot!(p, sol_step1, legend=:topright)
plot_height = 400 # hide
domain_height = 2.6 # hide
markersize = fluid_particle_spacing / (domain_height / plot_height) / 2 # hide
p = plot(tank.fluid, tank.boundary, labels=["fluid" "boundary"]; markersize) # hide
plot!(p, sol_step1, legend=:topright, dpi=200; markersize) # hide
savefig(p, "tut_rigid_body_fsi_step1.png"); # hide
nothing # hide
# ![Step 1](tut_rigid_body_fsi_step1.png)

# ## Step 2: With boundary model, without contact model
# Now, we introduce fluid-structure interaction. We define a `boundary_model` for each rigid body,
# which allows them to interact with the fluid. We also add the tank boundary to the simulation.
# However, we still don't use a `contact_model`, so the rigid bodies will not collide with the
# tank or each other.
# For rigid-body FSI, the rigid particles need two separate pieces of information:
#
# 1. their physical density, which is already stored in `square1` and `square2`
#    and determines the rigid-body mass and inertia,
# 2. a boundary model for the fluid coupling, which requires "hydrodynamic
#    masses" and "hydrodynamic densities" on the rigid-body surface.
#
# In this tutorial, we initialize the "hydrodynamic density" from the surrounding fluid density.
# See [the docs on dummy particles](@ref boundary_models) for a definition for these terms.

boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary; fluid_system=fluid_system,
                                             boundary_density_calculator=boundary_density_calculator)
boundary_system = WallBoundarySystem(tank.boundary, boundary_model)
nothing # hide

function rigid_body_boundary_model(shape)
    hydrodynamic_densities = fluid_density * ones(size(shape.density))
    hydrodynamic_masses = hydrodynamic_densities *
                          structure_particle_spacing^ndims(fluid_system)

    return BoundaryModelDummyParticles(hydrodynamic_densities, hydrodynamic_masses,
                                       state_equation=state_equation,
                                       boundary_density_calculator,
                                       fluid_smoothing_kernel,
                                       fluid_smoothing_length)
end

square1_boundary_model = rigid_body_boundary_model(square1)
square2_boundary_model = rigid_body_boundary_model(square2)
nothing # hide

rigid_body_system_1_step2 = RigidBodySystem(square1;
                                            boundary_model=square1_boundary_model,
                                            acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
rigid_body_system_2_step2 = RigidBodySystem(square2;
                                            boundary_model=square2_boundary_model,
                                            acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
nothing # hide

semi_step2 = Semidiscretization(fluid_system, boundary_system,
                                rigid_body_system_1_step2, rigid_body_system_2_step2)
ode_step2 = semidiscretize(semi_step2, tspan)
nothing # hide

info_callback = InfoCallback(interval=100) # hide
saving_callback_step2 = SolutionSavingCallback(dt=0.02, prefix="step2") # hide
callbacks_step2 = CallbackSet(info_callback, saving_callback_step2) # hide
nothing # hide

# ```julia
# sol_step2 = solve(ode_step2, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks)
# ```
sol_step2 = solve(ode_step2, RDPK3SpFSAL49(), save_everystep=false, abstol=1e-6, # hide
                  reltol=1e-4, dtmax=2e-3) # hide
nothing # hide

# In the plot, you can see the interaction between the fluid and the squares.
# However, the squares pass through the tank bottom and each other.
plot(sol_step2)
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_step2.png"); # hide
nothing # hide
# ![Step 2](tut_rigid_body_fsi_step2.png)

# ## Step 3: With contact model
# Finally, we add a `contact_model` to handle collisions between rigid bodies and between
# rigid bodies and the tank.

contact_model = RigidContactModel(; normal_stiffness=2.0e5,
                                  normal_damping=150.0,
                                  contact_distance=2.0 * structure_particle_spacing)
nothing # hide

rigid_body_system_1_step3 = RigidBodySystem(square1;
                                            boundary_model=square1_boundary_model,
                                            contact_model=contact_model,
                                            acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
rigid_body_system_2_step3 = RigidBodySystem(square2;
                                            boundary_model=square2_boundary_model,
                                            contact_model=contact_model,
                                            acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
nothing # hide

semi_step3 = Semidiscretization(fluid_system, boundary_system,
                                rigid_body_system_1_step3, rigid_body_system_2_step3)
ode_step3 = semidiscretize(semi_step3, tspan)
nothing # hide

info_callback = InfoCallback(interval=100) # hide
saving_callback_step3 = SolutionSavingCallback(dt=0.02, prefix="step3") # hide
callbacks_step3 = CallbackSet(info_callback, saving_callback_step3) # hide
nothing # hide

# ```julia
# sol_step3 = solve(ode_step3, RDPK3SpFSAL49(), save_everystep=false,
#                   callback=callbacks,  abstol=1e-6, reltol=1e-4, dtmax=2e-3)
# ```
sol_step3 = solve(ode_step3, RDPK3SpFSAL49(), abstol=1e-6, reltol=1e-4, dtmax=2e-3, # hide
                  save_everystep=false) # hide
nothing # hide

# The plot now shows the full simulation. The squares collide with the tank bottom and each other.
plot(sol_step3)
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_step3.png"); # hide
nothing # hide
# ![Step 3](tut_rigid_body_fsi_step3.png)

# ## Step 4: Different geometry
# The same setup can be used with different geometries. Here, we replace the squares with circles.

circle1_radius = 0.2
circle2_radius = 0.15

circle1_center = (0.4, 1.25)
circle2_center = (1.25, 1.30)

circle1 = SphereShape(structure_particle_spacing, circle1_radius, circle1_center,
                      square1_density, sphere_type=RoundSphere())
circle2 = SphereShape(structure_particle_spacing, circle2_radius, circle2_center,
                      square2_density, sphere_type=RoundSphere())

circle1 = apply_angular_velocity(circle1, square1_angular_velocity)
circle2 = apply_angular_velocity(circle2, square2_angular_velocity)
nothing # hide

# Let's visualize the new setup.
plot(tank.fluid, tank.boundary, circle1, circle2,
     labels=["fluid" "boundary" "circle 1" "circle 2"])
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_setup_circles.png"); # hide
nothing # hide
# ![initial setup with circles](tut_rigid_body_fsi_setup_circles.png)

circle1_boundary_model = rigid_body_boundary_model(circle1)
circle2_boundary_model = rigid_body_boundary_model(circle2)
nothing # hide

rigid_body_system_1_step4 = RigidBodySystem(circle1;
                                            boundary_model=circle1_boundary_model,
                                            contact_model=contact_model,
                                            acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
rigid_body_system_2_step4 = RigidBodySystem(circle2;
                                            boundary_model=circle2_boundary_model,
                                            contact_model=contact_model,
                                            acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
nothing # hide

semi_step4 = Semidiscretization(fluid_system, boundary_system,
                                rigid_body_system_1_step4, rigid_body_system_2_step4)
ode_step4 = semidiscretize(semi_step4, tspan)
nothing # hide

info_callback = InfoCallback(interval=100) # hide
saving_callback_step4 = SolutionSavingCallback(dt=0.02, prefix="step4") # hide
callbacks_step4 = CallbackSet(info_callback, saving_callback_step4) # hide
nothing # hide

# ```julia
# sol_step4 = solve(ode_step4, RDPK3SpFSAL49(), save_everystep=false,
#                   callback=callbacks,  abstol=1e-6, reltol=1e-4, dtmax=2e-3)
# ```
sol_step4 = solve(ode_step4, RDPK3SpFSAL49(), abstol=1e-6, reltol=1e-4, dtmax=2e-3, # hide
                  save_everystep=false) # hide
nothing # hide

# And here is the final plot with circles instead of squares.
plot(sol_step4)
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_step4.png"); # hide
nothing # hide
# ![Step 4](tut_rigid_body_fsi_step4.png)

# ## Next steps
#
# A natural extension is to add a layer of small rigid spheres on top of the initial water column.
# If we choose a density of `500.0`, the spheres are lighter than the surrounding fluid and should
# mostly float while interacting with the waves and the falling squares.
# The example file [`examples/fsi/falling_rotating_rigid_squares_2d.jl`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/examples/fsi/falling_rotating_rigid_squares_2d.jl)
# contains the same idea as a commented-out block.
#
# One possible implementation is:
small_sphere_radius = 0.08
small_sphere_density = 500.0
small_sphere_y = initial_fluid_size[2] + small_sphere_radius
small_sphere_x_positions = 0.25:(3 * small_sphere_radius):1.75

small_spheres = [SphereShape(structure_particle_spacing, small_sphere_radius,
                             (x, small_sphere_y), small_sphere_density,
                             sphere_type=RoundSphere())
                 for x in small_sphere_x_positions]

small_sphere_systems = [begin
                            sphere_boundary_model = rigid_body_boundary_model(sphere)
                            RigidBodySystem(sphere;
                                            boundary_model=sphere_boundary_model,
                                            contact_model=contact_model,
                                            acceleration=(0.0, -gravity),
                                            particle_spacing=structure_particle_spacing)
                        end
                        for sphere in small_spheres]

semi_next = Semidiscretization(fluid_system, boundary_system,
                               rigid_body_system_1_step3, rigid_body_system_2_step3,
                               small_sphere_systems...)

ode_step_next = semidiscretize(semi_next, tspan) #hide
sol_step_next = solve(ode_step_next, RDPK3SpFSAL49(), # hide
                      abstol=1e-6, reltol=1e-4, dtmax=2e-3, # hide
                      save_everystep=false) # hide
nothing # hide

plot(sol_step_next, legend=nothing) #hide
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_next.png"); # hide
nothing # hide
# ![next_step](tut_rigid_body_fsi_next.png)

#
# This modification is useful for studying many-body rigid-body FSI, including
# floating-particle rafts, repeated rigid-rigid contact, and wave-driven rearrangement.
#
# ### Loading Geometries from Files
#
# Instead of using predefined shapes like `RectangularShape` or `SphereShape`, you can load
# custom geometries from external files. For 2D, `TrixiParticles.jl` supports a simple
# ASCII format with the extension `.asc`.
#
# An `.asc` file should contain a list of 2D coordinates, with x and y values separated by a space,
# and one point per line. The points should form a closed polygon.
# `TrixiParticles.jl` includes some example files in the `examples/preprocessing/data` directory.
#
# Here is how you can load the `hexagon.asc` file from this directory, create a `ComplexShape` from it,
# and then use it in a simulation.
#
# Load the geometry from an .asc file.
file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "hexagon.asc")
loaded_geometry = load_geometry(file)
#
# Create a `ComplexShape` from the loaded geometry.
# We can specify the particle spacing, a starting position, and the density.
hexagon_density = 1500.0
hexagon_shape = ComplexShape(loaded_geometry, particle_spacing=2*structure_particle_spacing,
                             density=hexagon_density)

hexagon_shape.coordinates .*= (0.5, 0.5) # Scale the shape
hexagon_shape.coordinates .+= (1.0, 1.5) # Shift the shape to the desired position
hexagon_shape = TrixiParticles.@set hexagon_shape.particle_spacing = structure_particle_spacing

# Now, `hexagon_shape` can be used to create a `RigidBodySystem`
# just like the other shapes in this tutorial.
hexagon_boundary_model = rigid_body_boundary_model(hexagon_shape)

hexagon_system = RigidBodySystem(hexagon_shape;
                                 boundary_model=hexagon_boundary_model,
                                 contact_model=contact_model,
                                 acceleration=(0.0, -gravity),
                                 particle_spacing=structure_particle_spacing)
#
# You can then create a semidiscretization with this new system.
semi_hexagon = Semidiscretization(fluid_system, boundary_system, hexagon_system)
ode_step_hex = semidiscretize(semi_hexagon, (0.0, 0.4))

sol_step_hex = solve(ode_step_hex, RDPK3SpFSAL49(), abstol=1e-6, reltol=1e-5, dtmax=1e-3,
                     save_everystep=false)
plot(sol_step_hex, legend=nothing) #hide
plot!(dpi=200) # hide
savefig("tut_rigid_body_fsi_hex.png"); # hide
nothing # hide

# ![hexagon_step](tut_rigid_body_fsi_hex.png)

# This allows you to simulate FSI with arbitrary 2D shapes.
#
# ### Example `.asc` Files
#
# You can find the `.asc` files used in this tutorial and other examples in the
# [`examples/preprocessing/data`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/examples/preprocessing/data/)
# directory of the `TrixiParticles.jl` repository.
#
# Some files can be found in examples/preprocessing/data.
# To use them, you can use `pkgdir` as shown above, for example:
#
file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "triangle.asc")
loaded_geometry = load_geometry(file)
