using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSymplecticRK

# ==========================================================================================
# ==== Resolution
n_particles_y = 4

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

fin_length = 0.6
fin_thickness = 2e-3
real_thickness = 1e-3
real_modulus = 125e9
poisson_ratio = 0.3
flexural_rigidity = real_modulus * real_thickness^3 / (1 - poisson_ratio^2) / 12
modulus = 12 * (1 - poisson_ratio^2) * flexural_rigidity / fin_thickness^3

fiber_volume_fraction = 0.6
fiber_density = 1800.0
epoxy_density = 1250.0
density = fiber_volume_fraction * fiber_density +
          (1 - fiber_volume_fraction) * epoxy_density

clamp_radius = 0.05

tank_size = (0.8, 0.2)
center = (0.05, 0.1)
initial_fluid_size = tank_size
initial_velocity = (1.0, 0.0)

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = fin_thickness / (n_particles_y - 1)
fluid_particle_spacing = particle_spacing

smoothing_length_structure = sqrt(2) * particle_spacing
smoothing_length_fluid = 1.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

# Beam and clamped particles
length_clamp = round(Int, 0.01 / particle_spacing) * particle_spacing # m
n_particles_per_dimension = (round(Int, (length_clamp) / particle_spacing) + 2,# + n_particles_clamp_x,
                             n_particles_y)
shape_sampled = RectangularShape(particle_spacing, n_particles_per_dimension,
                                 (-length_clamp, 0.0), density=density, place_on_shell=true)
length_clamp = 0.0
n_particles_per_dimension = (round(Int, (fin_length + length_clamp) / particle_spacing) + 2,# + n_particles_clamp_x,
                             n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for structures.
# We therefore need to pass `place_on_shell=true`.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (-length_clamp, 0.0), density=density, place_on_shell=true)

fixed_particles = setdiff(shape_sampled, beam)

# structure = union(beam, fixed_particles)

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 3

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 6

fluid_density = 1000.0
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers,
                       faces=(false, false, true, true), velocity=initial_velocity)
# fluid = setdiff(tank.fluid, structure)

open_boundary_size = (fluid_particle_spacing * open_boundary_layers, tank_size[2])

min_coords_inlet = (-open_boundary_layers * fluid_particle_spacing, 0.0)
inlet = RectangularTank(fluid_particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers,
                        min_coordinates=min_coords_inlet,
                        faces=(false, false, true, true))

min_coords_outlet = (tank.fluid_size[1], 0.0)
outlet = RectangularTank(fluid_particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         min_coordinates=min_coords_outlet,
                         faces=(false, false, true, true))


NDIMS = ndims(tank.fluid)
n_buffer_particles = 10 * tank.n_particles_per_dimension[2]^(NDIMS - 1)


structure = union(beam, fixed_particles)
# Move the fin to the center of the tank
structure.coordinates .+= center .+ (fluid_particle_spacing / 2, fluid_particle_spacing / 2)

fluid = setdiff(tank.fluid, structure)

n_clamped_particles = nparticles(structure) - nparticles(beam)

# Movement function
frequency = 1.3 # Hz
amplitude = 0.18 # m
rotation_deg = 25 # degrees
rotation_phase_offset = 0.12 # periods
translation_vector = SVector(0.0, amplitude)
rotation_angle = rotation_deg * pi / 180

boundary_motion = OscillatingMotion2D(; frequency,
                                      translation_vector=SVector(0.0, amplitude),
                                      rotation_angle, rotation_center=center,
                                      rotation_phase_offset, ramp_up_tspan=(0.0, 0.5))

sound_speed = 40.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, background_pressure=0.0)

# ==========================================================================================
# ==== Structure
boundary_density_calculator = AdamiPressureExtrapolation()
viscosity_fluid = ViscosityAdami(nu=1e-4)
viscosity_fin = ViscosityAdami(nu=1e-4)

# For the FSI we need the hydrodynamic masses and densities in the structure boundary model
hydrodynamic_densites = fluid_density * ones(size(structure.density))
hydrodynamic_masses = hydrodynamic_densites * particle_spacing^2

boundary_model_structure = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                   hydrodynamic_masses,
                                                   state_equation=state_equation,
                                                   boundary_density_calculator,
                                                   smoothing_kernel, smoothing_length_fluid,
                                                   viscosity=viscosity_fin)

# k_structure = 1.0
# beta_structure = fluid_particle_spacing / particle_spacing
# boundary_model_structure = BoundaryModelMonaghanKajtar(k_structure, beta_structure,
#                                                    particle_spacing,
#                                                    hydrodynamic_masses)

structure_system = TotalLagrangianSPHSystem(structure, smoothing_kernel, smoothing_length_structure,
                                        modulus, poisson_ratio;
                                        n_clamped_particles, #clamped_particles_motion=boundary_motion,
                                        velocity_averaging=nothing,
                                        boundary_model=boundary_model_structure,
                                        viscosity=ArtificialViscosityMonaghan(alpha=0.01),
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

# ==========================================================================================
# ==== Fluid
fluid_density_calculator = ContinuityDensity()
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
# density_diffusion = DensityDiffusionAntuono(fluid, delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length_fluid, viscosity=viscosity_fluid,
                                           density_diffusion=density_diffusion,
                                           shifting_technique=ParticleShiftingTechnique(sound_speed_factor=0.2, v_max_factor=0.0),
                                        #    pressure_acceleration=tensile_instability_control,
                                           buffer_size=n_buffer_particles)
# fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
#                                            sound_speed, viscosity=ViscosityAdami(; nu),
#                                            transport_velocity=TransportVelocityAdami(10 * sound_speed^2 * fluid_density))

min_corner = minimum(tank.boundary.coordinates, dims=2) .- fluid_particle_spacing / 2
max_corner = maximum(tank.boundary.coordinates, dims=2) .+ fluid_particle_spacing / 2
min_corner = convert.(typeof(fluid_particle_spacing), min_corner)
max_corner = convert.(typeof(fluid_particle_spacing), max_corner)
periodic_box = PeriodicBox(; min_corner, max_corner)
open_boundary_system = nothing
wall = tank.boundary

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length_fluid)

boundary_system = WallBoundarySystem(wall, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(wall.coordinates, dims=2) .- fluid_particle_spacing / 2
max_corner = maximum(wall.coordinates, dims=2) .+ fluid_particle_spacing / 2
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box, cell_list,
                                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, boundary_system, open_boundary_system, structure_system; neighborhood_search,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
prefix = "simplified"
saving_callback = SolutionSavingCallback(dt=0.01, prefix=prefix)

split_cfl = 1.6
# SSPRK104 CFL = 2.5, 15k RHS evaluations
# CarpenterKennedy2N54 CFL = 1.6, 11k RHS evaluations
# RK4 CFL = 1.2, 12k RHS evaluations
# VerletLeapfrog CFL = 0.5, 6.75k RHS evaluations
# VelocityVerlet CFL = 0.5, 6.75k RHS evaluations
# DPRKN4 CFL = 1.7, 9k RHS evaluations

split_integration = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false), adaptive=false,
                                             stage_coupling=true,
                                             dt=1e-5, # This is overwritten by the stepsize callback
                                             callback=StepsizeCallback(cfl=split_cfl),
                                             maxiters=10^8)

fluid_cfl = 1.2
stepsize_callback = StepsizeCallback(cfl=fluid_cfl)

function total_volume(system::WeaklyCompressibleSPHSystem, data, t)
    return sum(data.mass ./ data.density)
end
function total_volume(system, data, t)
    return nothing
end
pp_cb = PostprocessCallback(; total_volume, interval=100,
                            filename="$(prefix)_total_volume", write_file_interval=50)

function plane_vtk(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return nothing
end
function plane_vtk(system::WeaklyCompressibleSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi_, t)
    resolution = fluid_particle_spacing / 6
    pvd = TrixiParticles.paraview_collection("out/$(prefix)_plane"; append=t > 0)
    interpolate_plane_2d_vtk(min_corner, max_corner, resolution,
                             semi_, semi_.systems[1], v_ode, u_ode, include_wall_velocity=true,
                             filename="$(prefix)_plane_$(round(Int, t * 1000))", pvd=pvd, t=t)
    TrixiParticles.vtk_save(pvd)
    return nothing
end
interpolate_cb = PostprocessCallback(; plane_vtk, dt=0.01, filename="plane")

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(),
                        stepsize_callback, split_integration, pp_cb, interpolate_cb)

dt_fluid = 1.25e-4
sol = solve(ode,
            # RDPK3SpFSAL35(),
            CarpenterKennedy2N54(williamson_condition=false),
            dt=dt_fluid, # This is overwritten by the stepsize callback
            # reltol=1e-5, abstol=1e-7,
            save_everystep=false, callback=callbacks, maxiters=10^8);
