# ==========================================================================================
# 2D Falling Water Column on an Elastic Beam (FSI)
#
# This example simulates a column of water falling under gravity and impacting
# an elastic beam, which is clamped at one end (cantilever).
# It demonstrates Fluid-Structure Interaction where the fluid deforms the structure.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
n_particles_y = 5

# Load setup from oscillating beam example
trixi_include(@__MODULE__, joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl"),
              thickness=0.05, n_particles_y=n_particles_y,
              sol=nothing) # Don't run simulation, only include the setup part

# Fluid resolution
fluid_particle_spacing = 3 * particle_spacing

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.525, 1.0125)
tank_size = (2.0, 4.0)

fluid_density = 1000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

fluid = RectangularShape(fluid_particle_spacing,
                         round.(Int, (initial_fluid_size ./ fluid_particle_spacing)),
                         (0.1, 0.2), density=fluid_density)

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.2 * fluid_particle_spacing
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Structure
k = gravity * initial_fluid_size[2]
spacing_ratio = fluid_particle_spacing / particle_spacing

# For the FSI we need the hydrodynamic masses and densities in the structure boundary model
hydrodynamic_densites = fluid_density * ones(size(structure.density))
hydrodynamic_masses = hydrodynamic_densites * particle_spacing^2

boundary_model = BoundaryModelMonaghanKajtar(k, spacing_ratio, particle_spacing,
                                             hydrodynamic_masses)

structure_system = TotalLagrangianSPHSystem(structure,
                                            smoothing_kernel, smoothing_length,
                                            material.E, material.nu,
                                            boundary_model=boundary_model,
                                            clamped_particles=1:nparticles(clamped_particles),
                                            acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, structure_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.005, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
