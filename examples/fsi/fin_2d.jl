using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSymplecticRK

include("fin_2d/packing.jl")
include("fin_2d/geometry.jl")
include("fin_2d/fitted_movement.jl")

function convert_ic(ic, T)
    return InitialCondition{ndims(ic)}(ic.coordinates, ic.velocity, ic.mass, ic.density,
                                      ic.pressure, T(ic.particle_spacing))
end

# ==========================================================================================
# ==== Resolution
n_particles_y = 4
parallelization_backend = PolyesterBackend()

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 3.0)

# Length of the blade from the attachment to the tip.
blade_length = 0.522

# The blade is artificially thickened to allow for a coarser resolution.
artificial_blade_thickness = 30e-3

# We simulate a 2D slice with this thickness in the unmodeled third dimension.
# Parts of the fin that are thinner than this will have their density and modulus
# scaled accordingly.
max_blade_width = 19e-2

# The (estimated) real material parameters of the blade.
real_modulus = 40e9
poisson_ratio = 0.3

# The (estimated) real modulus of the foot pocket including the foot.
real_modulus_foot_pocket = 5e5

# Simulate the blade clamped in the foot pocket or just the blade.
simulate_foot_pocket = true
if !simulate_foot_pocket
    packing = false
    @assert packing == false "Packing is only supported when simulating the foot pocket."
end

# Real blade thickness profile along the flexible blade:
# x = 0 is the attachment to the foot pocket, x = 1 is the blade tip.
function real_thickness(x_relative)
    real_thickness_at_attachment = 1.2e-3
    real_thickness_at_tip = 0.7e-3
    taper_length = 0.522

    # `p = 1` is a linear profile.
    p = 1

    # Clamp to use constant material properties for the clamped part of the blade.
    x_clamped = clamp(x_relative / taper_length, 0.0, 1.0)
    return real_thickness_at_tip +
           (1 - x_clamped)^p * (real_thickness_at_attachment - real_thickness_at_tip)
end

tank_size = (2.0, 1.5)
center = (tank_size[2] / 2, tank_size[2] / 2)
initial_fluid_size = tank_size
initial_velocity = (1.0, 0.0)

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = artificial_blade_thickness / (n_particles_y - 1)
fluid_particle_spacing = particle_spacing

smoothing_length_structure = sqrt(2) * particle_spacing
smoothing_length_fluid = 1.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

# When the foot pocket is simulated, extend the blade into the foot pocket to guarantee
# a good particle distribution. Without a foot pocket, clamp 1cm of the blade.
length_clamp = simulate_foot_pocket ? 0.3 : 0.01
length_clamp = round(Int, length_clamp / particle_spacing) * particle_spacing # m

n_particles_x = round(Int, (blade_length + length_clamp) / particle_spacing) + 1
n_particles_per_dimension = (n_particles_x, n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for structures.
# We therefore need to pass `place_on_shell=true`.
# The density is arbitrary, as density and mass will be overwritten later.
blade = RectangularShape(particle_spacing, n_particles_per_dimension,
                         (-length_clamp, -artificial_blade_thickness / 2),
                         density=1000.0, place_on_shell=true)

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 3

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 10

fluid_density = 1000.0
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers,
                       faces=(false, false, true, true), velocity=initial_velocity)

open_boundary_size = (fluid_particle_spacing * open_boundary_layers, tank_size[2])

min_coords_inlet = (-open_boundary_layers * fluid_particle_spacing, 0.0)
inlet = RectangularTank(fluid_particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers,
                        min_coordinates=min_coords_inlet,
                        velocity=initial_velocity,
                        faces=(false, false, true, true))

min_coords_outlet = (tank.fluid_size[1], 0.0)
outlet = RectangularTank(fluid_particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         min_coordinates=min_coords_outlet,
                         velocity=initial_velocity,
                         faces=(false, false, true, true))


NDIMS = ndims(tank.fluid)
n_buffer_particles = 20 * tank.n_particles_per_dimension[2]^(NDIMS - 1)

# ==========================================================================================
# ==== Packing
packing = false
if packing
    foot_pocket, fluid = sample_and_pack(particle_spacing, center, blade, tank.fluid;
                                         parallelization_backend)
    fin = union(blade, foot_pocket)
else
    foot_pocket = sample_foot_pocket(particle_spacing, center, blade)

    fin = union(blade, foot_pocket)
    fluid = setdiff(tank.fluid, fin)
end

structure = union(blade, foot_pocket)

# Make sure that no overlapping particles have been removed. This should've been
# handled by the `setdiff` calls above.
@assert nparticles(structure) == nparticles(foot_pocket) + nparticles(blade)

# Movement function (parameters chosen to match video)
frequency = 1.06 # Hz
amplitude = 0.24 # m
rotation_deg = 22 # degrees
rotation_phase_offset = 0.18 # periods
rotation_center = center
rotation_angle = rotation_deg * pi / 180

if simulate_foot_pocket
    boundary_motion = OscillatingMotion2D(; frequency,
                                          translation_vector=SVector(0.0, amplitude),
                                          rotation_angle, rotation_center,
                                          rotation_phase_offset, ramp_up_tspan=(0.0, 0.5))
else
    structure = blade
    fluid = setdiff(tank.fluid, structure)

    # Spectral fitting of the blade motion of a simulation with foot pocket at
    # `artificial_blade_thickness = 2e-3`.
    fin_motion_period_start = 1.0
    fin_translation_x_coefficients = (-0.01596664749937493, 0.00462449770756477, -0.0034775925966148923, -0.007881846267683089, 0.012091124429516563, -0.00043060598134243605, 0.0001588281500050652, 0.001958732740588549, -0.0029380178559097877)
    fin_translation_y_coefficients = (0.008610136532786644, 0.028255124555259675, 0.23332024824975928, -0.0007805090695376739, -0.00011718123538573341, 0.010291034757006896, -0.00600372750522388, 0.0009596100233938224, -0.00030304878026988764)
    fin_rotation_coefficients = (0.024284835522184255, -0.8076437567872873, 0.20922602730868908, 0.01014650968308979, -0.015348776152286305, 0.04418104964022995, 0.013649587768681403, -0.010565069131844747, -0.0026430850704781943)

    fitted_blade_movement = fitted_movement(frequency, fin_motion_period_start,
                                            fin_translation_x_coefficients,
                                            fin_translation_y_coefficients,
                                            fin_rotation_coefficients, center)
    boundary_motion = PrescribedMotion(fitted_blade_movement, Returns(true))
end

sound_speed = 60.0
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
                                                       smoothing_kernel,
                                                       smoothing_length_fluid,
                                                       viscosity=viscosity_fin)

# Compute mass, density and modulus.
apply_material_properties!(structure, simulate_foot_pocket)
modulus = artificial_modulus(structure, simulate_foot_pocket)
clamped_particles = clamped_structure_particles(structure, simulate_foot_pocket)

viscosity_structure = ArtificialViscosityMonaghan(alpha=1.0)
structure_system = TotalLagrangianSPHSystem(structure; smoothing_kernel,
                                            smoothing_length=smoothing_length_structure,
                                            young_modulus=modulus, poisson_ratio,
                                            clamped_particles,
                                            clamped_particles_motion=boundary_motion,
                                            boundary_model=boundary_model_structure,
                                            velocity_averaging=TrixiParticles.VelocityAveraging(time_constant=5e-4),
                                            viscosity=viscosity_structure,
                                            penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

# ==========================================================================================
# ==== Fluid
fluid_density_calculator = ContinuityDensity()
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

shifting_technique = ParticleShiftingTechnique(sound_speed_factor=0.2, v_max_factor=0.0)
fluid_system = WeaklyCompressibleSPHSystem(fluid; density_calculator=fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length=smoothing_length_fluid,
                                           viscosity=viscosity_fluid,
                                           density_diffusion, shifting_technique,
                                           pressure_acceleration=nothing,
                                           buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundaries
periodic = false
if periodic
    min_corner = minimum(tank.boundary.coordinates, dims=2) .- fluid_particle_spacing / 2
    max_corner = maximum(tank.boundary.coordinates, dims=2) .+ fluid_particle_spacing / 2
    min_corner = convert.(typeof(fluid_particle_spacing), min_corner)
    max_corner = convert.(typeof(fluid_particle_spacing), max_corner)
    periodic_box = PeriodicBox(; min_corner, max_corner)
    open_boundary_system = nothing
    wall = tank.boundary
else
    periodic_box = nothing

    open_boundary_model = BoundaryModelDynamicalPressureZhang()
    reference_velocity_in = SVector(1.0, 0.0)
    reference_pressure_in = 0.0
    reference_density_in = nothing
    boundary_type_in = InFlow()
    face_in = ([0.0, 0.0], [0.0, tank_size[2]])
    flow_direction = [1.0, 0.0]
    inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                          open_boundary_layers, density=fluid_density, particle_spacing,
                          reference_density=reference_density_in,
                          reference_pressure=reference_pressure_in,
                          reference_velocity=reference_velocity_in,
                          initial_condition=inlet.fluid, boundary_type=boundary_type_in)

    reference_velocity_out = SVector(1.0, 0.0)
    reference_pressure_out = nothing
    reference_density_out = nothing
    boundary_type_out = OutFlow()
    face_out = ([min_coords_outlet[1], 0.0], [min_coords_outlet[1], tank_size[2]])
    outflow = BoundaryZone(; boundary_face=face_out, face_normal=(-flow_direction),
                           open_boundary_layers, density=fluid_density, particle_spacing,
                           reference_density=reference_density_out,
                           reference_pressure=reference_pressure_out,
                           reference_velocity=reference_velocity_out,
                           initial_condition=outlet.fluid, boundary_type=boundary_type_out)

    open_boundary_system = OpenBoundarySystem(inflow, outflow; fluid_system,
                                              boundary_model=open_boundary_model,
                                              buffer_size=n_buffer_particles)

    wall = union(tank.boundary, inlet.boundary, outlet.boundary)
    min_corner = minimum(wall.coordinates, dims=2) .- 5 * fluid_particle_spacing
    max_corner = maximum(wall.coordinates, dims=2) .+ 5 * fluid_particle_spacing
end

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
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box, cell_list,
                                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, boundary_system, open_boundary_system,
                          structure_system; neighborhood_search,
                          parallelization_backend)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
solution_prefix = ""
saving_callback = SolutionSavingCallback(dt=1/120; prefix=solution_prefix)

split_cfl = 2.1
# CarpenterKennedy2N54 CFL = 2.1, 23M RHS evaluations
# RK4 CFL = 1.8, 21.6M RHS evaluations
# VerletLeapfrog fails non-deterministically even at CFL = 0.5
# VelocityVerlet fails non-deterministically even at CFL = 0.5

split_integration = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                             adaptive=false,
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
                            filename="$(solution_prefix)_total_volume", write_file_interval=50)

efficiency_interval = 100
mechanical_work_calculator = MechanicalWorkCalculator(semi.systems[4], semi)
thrust_calculator = ThrustCalculator(semi.systems[4], semi, direction=SVector(1.0, 0.0))
calculator_cb = PostprocessCallback(; mechanical_work_calculator, thrust_calculator,
                                    interval=efficiency_interval, write_file_interval=10,
                                    filename="$(solution_prefix)_efficiency")

# Reconstruct the motion of the blade centerline at its attachment
# (`x = 0` in blade coordinates) from the surrounding SPH particles.
blade_motion = TrixiParticles.tlsph_motion(semi.systems[4], semi, center)

blade_motion_cb = PostprocessCallback(; blade_motion, dt=1 / 120,
                                      write_file_interval=10,
                                      filename="$(solution_prefix)_blade_motion")

callbacks = CallbackSet(info_callback, saving_callback,
                        stepsize_callback, split_integration, pp_cb,
                        calculator_cb, blade_motion_cb,
                        UpdateCallback(), SortingCallback(interval=10_000))

dt_fluid = 1.25e-4
sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition=false),
            dt=dt_fluid, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks, maxiters=10^8);
