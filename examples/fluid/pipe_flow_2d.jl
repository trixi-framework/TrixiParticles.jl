# TODO: Description
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
domain_length_factor = 0.05

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Due to the dynamics at the inlets and outlets, it is recommended to use
# `open_boundary_layers > boundary_layers`
open_boundary_layers = 6

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
domain_size = (1.0, 0.4)

flow_direction = [1.0, 0.0]
reynolds_number = 100
const prescribed_velocity = 2.0

particle_spacing = domain_length_factor * domain_size[1]

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2])

fluid_density = 1000.0
pressure = 1000.0

sound_speed = 10 * prescribed_velocity

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, background_pressure=pressure)

pipe = RectangularTank(particle_spacing, domain_size, boundary_size, fluid_density,
                       pressure=pressure, n_layers=boundary_layers,
                       faces=(false, false, true, true))

# Shift pipe walls in negative x-direction for the inflow
pipe.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

n_buffer_particles = 4 * pipe.n_particles_per_dimension[2]

# ==========================================================================================
# ==== Fluid
smoothing_length = 3.0 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

nu = prescribed_velocity * domain_size[2] / reynolds_number
alpha = 8 * nu / (smoothing_length * sound_speed)

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(; alpha, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(pipe.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           buffer=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
function velocity_function(pos, t)
    return SVector(prescribed_velocity, 0.0) # SVector(0.5prescribed_velocity * sin(2pi * t) + prescribed_velocity, 0)
end

inflow = InFlow(; plane=([0.0, 0.0], [0.0, domain_size[2]]), flow_direction,
                open_boundary_layers, density=fluid_density, particle_spacing)

open_boundary_in = OpenBoundarySPHSystem(inflow, sound_speed; buffer=n_buffer_particles,
                                         reference_pressure=pressure,
                                         reference_velocity=velocity_function)

outflow = OutFlow(; plane=([domain_size[1], 0.0], [domain_size[1], domain_size[2]]),
                  flow_direction, open_boundary_layers, density=fluid_density,
                  particle_spacing)
open_boundary_out = OpenBoundarySPHSystem(outflow, sound_speed; buffer=n_buffer_particles,
                                          reference_pressure=pressure,
                                          reference_velocity=velocity_function)

# ==========================================================================================
# ==== Boundary

boundary_model = BoundaryModelDummyParticles(pipe.boundary.density, pipe.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             #viscosity=ViscosityAdami(nu=1e-4),
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(pipe.boundary, boundary_model)

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

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);