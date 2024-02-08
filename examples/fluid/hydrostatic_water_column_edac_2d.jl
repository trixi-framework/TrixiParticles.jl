using TrixiParticles

# Import the setup from `hydrostatic_water_column_2d.jl`
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              fluid_particle_spacing=0.05, initial_fluid_size=(1.0, 0.9),
              tank_size=(1.0, 1.0), smoothing_kernel=SchoenbergQuinticSplineKernel{2}(),
              sol=nothing) # Overwrite `sol` assignment to skip time integration

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.0)

# ==========================================================================================
# ==== Fluid
alpha = 0.02
viscosity = ViscosityAdami(nu=alpha * smoothing_length * sound_speed / 8)

fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(),
            save_everystep=false, callback=callbacks);
