# TODO: Description
using TrixiParticles
using OrdinaryDiffEq

wcsph = false

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.05

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

open_boundary_cols = 5

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

# Boundary geometry
pipe_radius = 2.0
pipe_radius_inner = 1.0
prescribed_velocity = (0.0, 2.0)

reynolds_number = 1000.0
fluid_density = 1000.0
pressure = wcsph ? 10_000.0 : 100_000.0
sound_speed = 10 * maximum(prescribed_velocity)

pipe_size = (pipe_radius - pipe_radius_inner, pipe_radius)
pipe_coords_min = (-(boundary_layers * particle_spacing + pipe_radius), 0.0)
pipe_coords_max = (boundary_layers * particle_spacing + 2pipe_radius, pipe_size[2])

pipe_in = RectangularTank(particle_spacing, pipe_size, pipe_size, fluid_density;
                          n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                          faces=(true, true, false, false), pressure=pressure)
pipe_out = RectangularTank(particle_spacing, pipe_size, pipe_size, fluid_density;
                           n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                           min_coordinates=(pipe_radius + pipe_radius_inner, 0.0),
                           faces=(true, true, false, false), pressure=pressure)

fluid_curvature = SphereShape(particle_spacing, pipe_radius, pressure=pressure,
                              (pipe_radius, pipe_size[2] + 0.5particle_spacing),
                              fluid_density, sphere_type=RoundSphere(end_angle=pi),
                              n_layers=pipe_in.n_particles_per_dimension[1])

pipe_curvature = SphereShape(particle_spacing,
                             pipe_radius + particle_spacing * boundary_layers,
                             (pipe_radius, pipe_size[2] + 0.5particle_spacing),
                             fluid_density, sphere_type=RoundSphere(end_angle=pi))

pipe = union(pipe_out.boundary, pipe_in.boundary, setdiff(pipe_curvature, fluid_curvature))
fluid = union(pipe_out.fluid, pipe_in.fluid, fluid_curvature)

n_buffer_particles = 20 * pipe_in.n_particles_per_dimension[1]

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

if wcsph
    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
    state_equation = StateEquationCole(sound_speed, 7, fluid_density, pressure)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity=viscosity,
                                               buffer=n_buffer_particles)
else
    nu = maximum(prescribed_velocity) * (pipe_radius - pipe_radius_inner) / reynolds_number
    viscosity = ViscosityAdami(; nu) #alpha * smoothing_length * sound_speed / 8)

    fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                               sound_speed, viscosity=viscosity,
                                               transport_velocity=TransportVelocityAdami(pressure),
                                               buffer=n_buffer_particles)
end
# ==========================================================================================
# ==== Open Boundary
open_boundary_length = open_boundary_cols * particle_spacing
open_boundary_size = (pipe_size[1], open_boundary_length)

zone_origin_in = (0.0, -open_boundary_length)
zone_origin_out = (pipe_radius + pipe_radius_inner, 0.0)

inflow = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density; n_layers=boundary_layers,
                         init_velocity=prescribed_velocity, pressure=pressure,
                         min_coordinates=(0.0, -open_boundary_length),
                         spacing_ratio=spacing_ratio, faces=(true, true, false, false))
outflow = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                          min_coordinates=(pipe_radius + pipe_radius_inner,
                                           -open_boundary_length),
                          fluid_density; n_layers=boundary_layers, pressure=pressure,
                          spacing_ratio=spacing_ratio, faces=(true, true, false, false))

zone_plane_in = ([0.0; 0.0], [pipe_size[1]; 0.0])
zone_plane_out = ([pipe_radius + pipe_radius_inner; -open_boundary_length],
                  [pipe_radius + pipe_radius_inner + pipe_size[1]; -open_boundary_length])

open_boundary_in = OpenBoundarySPHSystem(inflow.fluid, InFlow(), fluid_system,
                                         flow_direction=(0.0, 1.0),
                                         zone_width=open_boundary_length,
                                         zone_plane_min_corner=[0.0, 0.0],
                                         zone_plane_max_corner=[pipe_size[1], 0.0],
                                         buffer=n_buffer_particles)

zone_plane_min_corner = [pipe_radius + pipe_radius_inner, 0.0]
zone_plane_max_corner = [pipe_radius + pipe_radius_inner + pipe_size[1], 0.0]
v_x(position, t) = prescribed_velocity[1]
v_y_out(position, t) = -prescribed_velocity[2]

open_boundary_out = OpenBoundarySPHSystem(outflow.fluid, OutFlow(), fluid_system,
                                          flow_direction=(0.0, -1.0),
                                          velocity_function=(v_x, v_y_out),
                                          zone_width=open_boundary_length,
                                          zone_plane_min_corner=zone_plane_min_corner,
                                          zone_plane_max_corner=zone_plane_max_corner,
                                          buffer=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
boundary = union(pipe, inflow.boundary, outflow.boundary)
boundary_density_calculator = AdamiPressureExtrapolation()
state_equation = wcsph ? state_equation : nothing

boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                             boundary_density_calculator,
                                             state_equation=state_equation,
                                             #viscosity=viscosity,
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

info_callback = InfoCallback(interval=50)
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
