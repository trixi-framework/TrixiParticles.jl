# TODO: Description
using TrixiParticles
using OrdinaryDiffEq
wcsph = false

# ==========================================================================================
# ==== Resolution
domain_length_factor = 0.05

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

open_boundary_cols = 5

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
domain_length = 1.0
domain_width = 0.4
reynolds_number = 100

particle_spacing = domain_length_factor * domain_length

fluid_density = 1000.0
pressure = wcsph ? 10_000.0 : 100_000.0

prescribed_velocity = (4.0, 0.0)

sound_speed = 10 * maximum(prescribed_velocity)

pipe = RectangularTank(particle_spacing, (domain_length, domain_width),
                       (domain_length + particle_spacing * open_boundary_cols,
                        domain_width), fluid_density, pressure=pressure,
                       loop_order_fluid=:x_first,
                       #init_velocity=prescribed_velocity,
                       n_layers=3, spacing_ratio=1, faces=(false, false, true, true))

n_particles_y = Int(floor(domain_width / particle_spacing))
n_buffer_particles = 4 * n_particles_y

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

if wcsph
    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
    state_equation = StateEquationCole(sound_speed, 7, fluid_density, pressure)

    fluid_system = WeaklyCompressibleSPHSystem(pipe.fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity=viscosity,
                                               buffer=n_buffer_particles)
else
    nu = maximum(prescribed_velocity) * domain_length / reynolds_number
    viscosity = ViscosityAdami(; nu) #alpha * smoothing_length * sound_speed / 8)

    fluid_system = EntropicallyDampedSPHSystem(pipe.fluid, smoothing_kernel,
                                               smoothing_length,
                                               sound_speed, viscosity=viscosity,
                                               transport_velocity=TransportVelocityAdami(pressure),
                                               buffer=n_buffer_particles)
end
# ==========================================================================================
# ==== Open Boundary
open_boundary_length = particle_spacing * open_boundary_cols
open_boundary_size = (open_boundary_length, domain_width)

inflow = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density; n_layers=boundary_layers,
                         init_velocity=prescribed_velocity, pressure=pressure,
                         min_coordinates=(-open_boundary_length, 0.0),
                         spacing_ratio=spacing_ratio,
                         loop_order_fluid=:x_first,
                         faces=(false, false, true, true))
outflow = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                          fluid_density; n_layers=boundary_layers,
                          loop_order_fluid=:x_first, pressure=pressure,
                          min_coordinates=(domain_length, 0.0), spacing_ratio=spacing_ratio,
                          faces=(false, false, true, true))

open_boundary_in = OpenBoundarySPHSystem(inflow.fluid, InFlow(), fluid_system,
                                         flow_direction=(1.0, 0.0),
                                         zone_width=open_boundary_length,
                                         zone_plane_min_corner=[0.0, 0.0],
                                         zone_plane_max_corner=[0.0, domain_width],
                                         buffer=n_buffer_particles)

v_x(p, t) = prescribed_velocity[1]
v_y(p, t) = prescribed_velocity[2]

open_boundary_out = OpenBoundarySPHSystem(outflow.fluid, OutFlow(), fluid_system,
                                          flow_direction=(1.0, 0.0),
                                          velocity_function=(v_x, v_y),
                                          zone_width=open_boundary_length,
                                          zone_plane_min_corner=[domain_length, 0.0],
                                          zone_plane_max_corner=[domain_length,
                                              domain_width],
                                          buffer=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
boundary = union(pipe.boundary, inflow.boundary, outflow.boundary)

state_equation = wcsph ? state_equation : nothing
boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             #viscosity=ViscosityAdami(nu=1e-4),
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system,
                          open_boundary_in,
                          open_boundary_out,
                          boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback, UpdateEachTimeStep())

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
