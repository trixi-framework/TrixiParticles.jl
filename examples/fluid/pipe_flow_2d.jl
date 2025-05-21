# ==========================================================================================
# 2D Pipe Flow Simulation with Open Boundaries (Inflow/Outflow)
#
# This example simulates fluid flow through a 2D pipe (channel) with an inflow
# boundary condition on one end and an outflow boundary condition on the other.
# Solid walls form the top and bottom of the pipe.
# The simulation demonstrates the use of open boundary conditions in TrixiParticles.jl.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

particle_spacing = 0.05
boundary_layers = 4

# Open boundary layers (inflow/outflow buffer zones)
# Recommended: `open_boundary_layers > boundary_layers` due to dynamics at open boundaries.
open_boundary_layers = 8

# Simulation time span
tspan = (0.0, 2.0)

# Choose fluid formulation: false for EDSPH, true for WCSPH
use_wcsph_formulation = false

# ------------------------------------------------------------------------------
# Experiment Setup:
# ------------------------------------------------------------------------------

# Physical domain size (length, height of the actual pipe segment)
pipe_domain_size = (1.0, 0.4)

# Flow properties
flow_direction_vector = SVector(1.0, 0.0) # Flow from left to right
reynolds_number = 100.0

# Prescribed inflow velocity magnitude (constant in this example).
# Made `const` for potential use in `velocity_function_inflow`.
const prescribed_inflow_velocity_magnitude = 2.0

# Computational domain size, including buffer regions for open boundaries.
# The boundary_size defines the extent of the `RectangularTank` object used
# to generate initial particles for fluid and solid walls.
computational_boundary_width = pipe_domain_size[1] +
                               2 * particle_spacing * open_boundary_layers
computational_boundary_height = pipe_domain_size[2]
computational_boundary_size = (computational_boundary_width, computational_boundary_height)

# Fluid properties
fluid_density_ref = 1000.0 # Reference density (kg/m^3)
# Background pressure can be important for stability with open boundaries,
# especially to prevent excessive suction at the outflow.
background_pressure = 1000.0 # Pa
sound_speed = 20 * prescribed_inflow_velocity_magnitude

# State equation will be defined later based on `use_wcsph_formulation`.
state_equation_fluid = nothing

# Create initial particles for the pipe (fluid and solid walls).
# `RectangularTank` generates particles within `computational_boundary_size`.
# The actual fluid domain is `pipe_domain_size`.
# Solid walls are only at the top and bottom.
pipe_particles_setup = RectangularTank(particle_spacing,
                                       pipe_domain_size, # Fluid fills this part initially
                                       computational_boundary_size, # Extent for particle generation
                                       fluid_density_ref,
                                       pressure=background_pressure, # Used for EDSPH/TransportVelocity or WCSPH background
                                       n_layers=boundary_layers,
                                       faces=(false, false, true, true)) # No solid left/right walls

# Shift the solid wall boundary particles to the left to align with the inflow plane.
# This ensures the solid walls start where the inflow boundary zone begins.
pipe_particles_setup.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

# Buffer size for SPH systems that involve particle creation/deletion (e.g., open boundaries).
# Estimate based on the number of particles in a cross-section of the open boundary.
num_particles_height = round(Int, pipe_domain_size[2] / particle_spacing)
# Factor of 4 is a heuristic for buffer capacity.
sph_system_buffer_size = 4 * num_particles_height

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
kinematic_viscosity_nu = prescribed_inflow_velocity_magnitude * pipe_domain_size[2] /
                         reynolds_number

if use_wcsph_formulation
    state_equation_fluid = StateEquationCole(sound_speed=sound_speed,
                                             reference_density=fluid_density_ref,
                                             exponent=1,
                                             background_pressure=background_pressure)
    # Artificial viscosity for WCSPH
    alpha_monaghan = 8 * kinematic_viscosity_nu / (smoothing_length * sound_speed)
    viscosity_model = ArtificialViscosityMonaghan(alpha=alpha_monaghan, beta=0.0)

    fluid_system = WeaklyCompressibleSPHSystem(pipe_particles_setup.fluid,
                                               fluid_density_calculator,
                                               state_equation_fluid,
                                               smoothing_kernel, smoothing_length,
                                               viscosity=viscosity_model,
                                               buffer_size=sph_system_buffer_size,
                                               reference_particle_spacing=particle_spacing)
else
    viscosity_model = ViscosityAdami(nu=kinematic_viscosity_nu)

    fluid_system = EntropicallyDampedSPHSystem(pipe_particles_setup.fluid,
                                               smoothing_kernel, smoothing_length,
                                               sound_speed,
                                               viscosity=viscosity_model,
                                               density_calculator=fluid_density_calculator,
                                               buffer_size=sph_system_buffer_size,
                                               reference_particle_spacing=particle_spacing)
end

# ------------------------------------------------------------------------------
# Open Boundary System Setup
# ------------------------------------------------------------------------------

# Velocity profile function for inflow/outflow boundaries.
# Can be time-dependent or position-dependent (parabolic, etc.).
# Here, a constant velocity profile is used.
function velocity_function_open_boundary(position, time)
    # Example for time-dependent inflow:
    # return SVector(0.5 * prescribed_inflow_velocity_magnitude * sin(2 * pi * time) +
    #                prescribed_inflow_velocity_magnitude, 0.0)
    return SVector(prescribed_inflow_velocity_magnitude, 0.0)
end

# Using Lastiwka model for open boundaries (see Lastiwka et al., 2009)
open_boundary_model_type = BoundaryModelLastiwka()

# Inflow Boundary
inflow_plane_start = [0.0, 0.0]
inflow_plane_end = [0.0, pipe_domain_size[2]]
inflow_plane = (inflow_plane_start, inflow_plane_end)
inflow_plane_normal = flow_direction_vector # Points into the domain
inflow_boundary_zone = BoundaryZone(plane=inflow_plane,
                                    plane_normal=inflow_plane_normal,
                                    open_boundary_layers=open_boundary_layers,
                                    density=fluid_density_ref,
                                    particle_spacing=particle_spacing,
                                    boundary_type=InFlow())

inflow_system = OpenBoundarySPHSystem(inflow_boundary_zone,
                                      fluid_system=fluid_system, # Link to the main fluid system
                                      boundary_model=open_boundary_model_type,
                                      buffer_size=sph_system_buffer_size,
                                      reference_density=fluid_density_ref,
                                      reference_pressure=background_pressure,
                                      reference_velocity=velocity_function_open_boundary)

# Outflow Boundary
outflow_plane_start = [pipe_domain_size[1], 0.0]
outflow_plane_end = [pipe_domain_size[1], pipe_domain_size[2]]
outflow_plane = (outflow_plane_start, outflow_plane_end)
outflow_plane_normal = -flow_direction_vector # Points out of the domain
outflow_boundary_zone = BoundaryZone(plane=outflow_plane,
                                     plane_normal=outflow_plane_normal,
                                     open_boundary_layers=open_boundary_layers,
                                     density=fluid_density_ref,
                                     particle_spacing=particle_spacing,
                                     boundary_type=OutFlow())

outflow_system = OpenBoundarySPHSystem(outflow_boundary_zone,
                                       fluid_system=fluid_system,
                                       boundary_model=open_boundary_model_type,
                                       buffer_size=sph_system_buffer_size,
                                       reference_density=fluid_density_ref,
                                       reference_pressure=background_pressure,
                                       reference_velocity=velocity_function_open_boundary)

# ------------------------------------------------------------------------------
# Solid Wall Boundary System Setup
# ------------------------------------------------------------------------------

# Viscosity model for wall-fluid interaction (e.g., for no-slip).
# Using a small Adami viscosity for slight damping at walls.
wall_viscosity_model = ViscosityAdami(nu=1e-4)
boundary_density_calculator_walls = AdamiPressureExtrapolation()

solid_wall_boundary_model = BoundaryModelDummyParticles(pipe_particles_setup.boundary.density,
                                                        pipe_particles_setup.boundary.mass,
                                                        boundary_density_calculator_walls,
                                                        state_equation=state_equation_fluid,
                                                        smoothing_kernel,
                                                        smoothing_length,
                                                        viscosity=wall_viscosity_model,
                                                        reference_particle_spacing=particle_spacing)

solid_wall_system = BoundarySPHSystem(pipe_particles_setup.boundary,
                                      solid_wall_boundary_model)

# ------------------------------------------------------------------------------
# Simulation Setup:
# ------------------------------------------------------------------------------

semi = Semidiscretization(fluid_system, inflow_system, outflow_system, solid_wall_system,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# `UpdateCallback` is crucial for open boundaries to manage particle buffers.
update_callback = UpdateCallback()
extra_callback = nothing # Placeholder for other potential callbacks

callbacks = CallbackSet(info_callback, saving_callback, update_callback, extra_callback)

# ------------------------------------------------------------------------------
# Simulation:
# ------------------------------------------------------------------------------

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
