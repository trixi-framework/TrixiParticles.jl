# ==========================================================================================
# 2D Pipe Flow Simulation with Open Boundaries (Inflow/Outflow)
#
# This example simulates fluid flow through a 2D pipe (channel) with an inflow
# boundary condition on one end and an outflow boundary condition on the other.
# Solid walls form the top and bottom of the pipe.
# The simulation demonstrates the use of open boundary conditions in TrixiParticles.jl.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.02

spacing_ratio = 2
min_spacing = particle_spacing / spacing_ratio
max_spacing = particle_spacing

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 6 * spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
domain_size = (1.0, 0.4)

reynolds_number = 100
const prescribed_velocity = (1.0, 0.0)
flow_direction = [1.0, 0.0]

open_boundary_size = (min_spacing * open_boundary_layers, domain_size[2])

fluid_density = 1000.0

sound_speed = 10 * maximum(abs.(prescribed_velocity))

# The pipe walls and the open boundaries have the minimum spacing of the refinement.
# They hold the fluid next to them at this spacing.
pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density;
                       n_layers=boundary_layers, velocity=prescribed_velocity,
                       faces=(false, false, true, true),
                       coordinates_eltype=Float64, spacing_ratio)

min_coords_inlet = (-open_boundary_layers * min_spacing, 0.0)
inlet = RectangularTank(min_spacing, open_boundary_size, open_boundary_size,
                        fluid_density; n_layers=boundary_layers,
                        min_coordinates=min_coords_inlet,
                        faces=(false, false, true, true),
                        coordinates_eltype=Float64)

min_coords_outlet = (pipe.fluid_size[1], 0.0)
outlet = RectangularTank(min_spacing, open_boundary_size, open_boundary_size,
                         fluid_density; n_layers=boundary_layers,
                         min_coordinates=min_coords_outlet,
                         faces=(false, false, true, true),
                         coordinates_eltype=Float64)

NDIMS = ndims(pipe.fluid)

n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^(NDIMS - 1)

# ==========================================================================================
# ==== Fluid
smoothing_length_ratio = 1.5
smoothing_length = smoothing_length_ratio * particle_spacing
smoothing_kernel = WendlandC2Kernel{NDIMS}()

# fluid_density_calculator = SummationDensity()
fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = maximum(prescribed_velocity) * domain_size[2] / reynolds_number

viscosity = ViscosityAdami(nu=kinematic_viscosity)

state_equation = nothing
# ==========================================================================================
# ==== Refinement
spatial_criterion = TrixiParticles.SpatialRefinementCriterion()
inlet_logic = (system, particle, v, u,
               semi) -> begin
    x_min = 0.0
    x_max = 0.15
    y_min = 0.0
    y_max = 0.4

    pos = TrixiParticles.current_coords(u, system, particle)
    if (x_min < pos[1] < x_max) && (y_min < pos[2] < y_max)
        return min_spacing
    else
        return max_spacing
    end
end

outlet_logic = (system, particle, v, u,
                semi) -> begin
    x_min = 0.85
    x_max = 1.0
    y_min = 0.0
    y_max = 0.4

    pos = TrixiParticles.current_coords(u, system, particle)
    if (x_min < pos[1] < x_max) && (y_min < pos[2] < y_max)
        return min_spacing
    else
        return max_spacing
    end
end

inlet_criterion = TrixiParticles.SolutionRefinementCriterion(inlet_logic)
outlet_criterion = TrixiParticles.SolutionRefinementCriterion(outlet_logic)
refinement_criteria = (spatial_criterion, inlet_criterion, outlet_criterion)

n_buffer_refinement_particles = 1000
n_buffer_particles += n_buffer_refinement_particles

# With `spacing_ratio=1.2`, it takes `log(2) / log(1.2) ≈ 4` neighborhoods
# to grade from `min_spacing` to `max_spacing`
refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                               min_spacing, max_spacing,
                                               smoothing_length_factor=smoothing_length_ratio,
                                               reference_density=fluid_density,
                                               refinement_criteria)

fluid_system = EntropicallyDampedSPHSystem(pipe.fluid; smoothing_kernel,
                                           smoothing_length, sound_speed, viscosity,
                                           density_calculator=fluid_density_calculator,
                                           shifting_technique=ParticleShiftingTechnique(),
                                           buffer_size=n_buffer_particles,
                                           particle_refinement=refinement)
# ==========================================================================================
# ==== Open Boundary

function velocity_function2d(pos, t)
    # Use this for a time-dependent inflow velocity
    # return SVector(0.5prescribed_velocity * sin(2pi * t) + prescribed_velocity, 0)

    return SVector(prescribed_velocity)
end
open_boundary_spacing = min_spacing
open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())

reference_velocity_in = velocity_function2d
reference_pressure_in = nothing
# Prescribe the density in the open boundaries. Extrapolating it from the fluid
# is unstable when particles of different masses enter the boundary zones.
reference_density_in = fluid_density
boundary_type_in = InFlow()
face_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density,
                      particle_spacing=open_boundary_spacing,
                      reference_density=reference_density_in,
                      reference_pressure=reference_pressure_in,
                      reference_velocity=reference_velocity_in,
                      initial_condition=inlet.fluid, boundary_type=boundary_type_in)

reference_velocity_out = nothing
reference_pressure_out = 0.0
reference_density_out = fluid_density
boundary_type_out = OutFlow()
face_out = ([min_coords_outlet[1], 0.0], [min_coords_outlet[1], domain_size[2]])
outflow = BoundaryZone(; boundary_face=face_out, face_normal=(-flow_direction),
                       open_boundary_layers, density=fluid_density,
                       particle_spacing=open_boundary_spacing,
                       reference_density=reference_density_out,
                       reference_pressure=reference_pressure_out,
                       reference_velocity=reference_velocity_out,
                       initial_condition=outlet.fluid, boundary_type=boundary_type_out)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
wall = union(pipe.boundary, inlet.boundary, outlet.boundary)
viscosity_boundary = viscosity
min_smoothing_length = smoothing_length_ratio * min_spacing

boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             min_smoothing_length; state_equation,
                                             viscosity=viscosity_boundary)

boundary_system = WallBoundarySystem(wall, boundary_model)

# ==========================================================================================
# ==== Simulation
padding = 4 * smoothing_length
min_corner = minimum(wall.coordinates .- padding, dims=2)
max_corner = maximum(wall.coordinates .+ padding, dims=2)

nhs = GridNeighborhoodSearch{NDIMS}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, boundary_system,
                          neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

extra_callback = nothing

refinement_callback = TrixiParticles.ParticleRefinementCallback(interval=10)

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback,
                        refinement_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
