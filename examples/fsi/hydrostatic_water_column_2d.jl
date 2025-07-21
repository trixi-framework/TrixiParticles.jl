############################################################################################
# Case "Elastic plate under a hydrostatic water column" as described in
# "A fluid–structure interaction model for free-surface flows and flexible structures
# using smoothed particle hydrodynamics on a GPU" by J. O'Connor and B.D. Rogers
# published in Journal of Fluids and Structures
# https://doi.org/10.1016/j.jfluidstructs.2021.103312
############################################################################################

using TrixiParticles
using OrdinaryDiffEq
using JSON

# ============================================================================
# Options
# ============================================================================
use_edac = true  # Use EDAC or WCSPH

# ============================================================================
# Experiment Parameters
# ============================================================================
# The paper uses 5 to 40 (3 for short runtime in CI)
n_particles_plate_y = 3
boundary_layers = 3
spacing_ratio = 1
gravity = 9.81
# Recommended: run at least to 0.5 (paper runs to 1.0); using 0.3 for speed in CI
tspan = (0.0, 0.3)

initial_fluid_size = (1.0, 2.0)
plate_size = (1.0, 0.05)

fluid_density = 1000.0
solid_density = 2700.0

E = 67.5e9
nu = 0.3
sound_speed = 50

# Particle spacings (solid and fluid share same spacing)
solid_particle_spacing = plate_size[2] / (n_particles_plate_y - 1)
fluid_particle_spacing = solid_particle_spacing
boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# Analytical solution (constant)
D = E * plate_size[2]^3 / (12 * (1 - nu^2))
analytical_value = -0.0026 * gravity *
                   (fluid_density * initial_fluid_size[2] +
                    solid_density * plate_size[2]) / D

# ============================================================================
# ==== Geometry Definitions: Tank, Beam, and Fixed Particles
# ============================================================================
n_particles_plate_x = round(Int, plate_size[1] / solid_particle_spacing + 1)
n_particles_per_dimension = (n_particles_plate_x, n_particles_plate_y)

plate = RectangularShape(solid_particle_spacing, n_particles_per_dimension,
                         (0.0, -plate_size[2]), density=solid_density, tlsph=true)

left_wall = RectangularShape(solid_particle_spacing, (3, n_particles_plate_y),
                             (-3 * solid_particle_spacing, -plate_size[2]),
                             density=solid_density, tlsph=true)
right_wall = RectangularShape(solid_particle_spacing, (3, n_particles_plate_y),
                              (plate_size[1] + solid_particle_spacing,
                               -plate_size[2]), density=solid_density, tlsph=true)
fixed_particles = union(left_wall, right_wall)

solid_geometry = union(plate, fixed_particles)

# ============================================================================
# Smoothing Kernel, Boundary, and Related Quantities
# ============================================================================
smoothing_kernel = WendlandC2Kernel{2}()
smoothing_length_solid = sqrt(2) * solid_particle_spacing

# Note: Setting this to something else than the solid particle spacing results in a larger error
smoothing_length_fluid = sqrt(2) * fluid_particle_spacing

hydrodynamic_densities = fluid_density * ones(size(solid_geometry.density))
hydrodynamic_masses = hydrodynamic_densities * solid_particle_spacing^ndims(solid_geometry)
boundary_density_calculator = AdamiPressureExtrapolation()

# ============================================================================
# Run Simulations
# ============================================================================

local_state_equation = use_edac ? nothing :
                       StateEquationCole(; sound_speed, reference_density=fluid_density,
                                         exponent=7, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, (plate_size[1], 3.0),
                       min_coordinates=(0.0, fluid_particle_spacing / 2),
                       fluid_density, n_layers=boundary_layers,
                       spacing_ratio=spacing_ratio,
                       faces=(true, true, false, false),
                       acceleration=(0.0, -gravity),
                       state_equation=local_state_equation)

if use_edac
    fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel,
                                               smoothing_length_fluid, sound_speed,
                                               acceleration=(0.0, -gravity),
                                               correction=ShepardKernelCorrection(),
                                               source_terms=SourceTermDamping(;
                                                                              damping_coefficient=0.05))
else
    fluid_density_calculator = ContinuityDensity()
    density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                               local_state_equation, smoothing_kernel,
                                               smoothing_length_fluid,
                                               density_diffusion=density_diffusion,
                                               acceleration=(0.0, -gravity),
                                               source_terms=SourceTermDamping(;
                                                                              damping_coefficient=0.05))
end

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=local_state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length_fluid)
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)
boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densities,
                                                   hydrodynamic_masses,
                                                   state_equation=local_state_equation,
                                                   boundary_density_calculator,
                                                   smoothing_kernel,
                                                   smoothing_length_solid)
solid_system = TotalLagrangianSPHSystem(solid_geometry, smoothing_kernel,
                                        smoothing_length_solid,
                                        E, nu, boundary_model=boundary_model_solid,
                                        n_fixed_particles=nparticles(fixed_particles),
                                        acceleration=(0.0, -gravity))

# semi = Semidiscretization(solid_system, fluid_system, boundary_system)

min_corner = [-1; -1]
max_corner = [3; 3]
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=ParallelUpdate();
                                                cell_list)
semi = Semidiscretization(solid_system, fluid_system, boundary_system,
                          neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

# for overwriting via trixi_include
extra_callback = nothing

info_callback = InfoCallback(interval=500)
saving_callback = SolutionSavingCallback(dt=0.1, prefix="")
callbacks = CallbackSet(info_callback, saving_callback, extra_callback)
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks)
