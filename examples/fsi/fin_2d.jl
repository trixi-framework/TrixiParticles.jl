using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSymplecticRK

function convert_ic(ic, T)
    return InitialCondition{ndims(ic)}(ic.coordinates, ic.velocity, ic.mass, ic.density,
                                      ic.pressure, T(ic.particle_spacing))
end

# ==========================================================================================
# ==== Resolution
n_particles_y = 4

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

fin_length = 0.502
fin_thickness = 30e-3
blade_width = 19e-2
real_modulus = 125e9
poisson_ratio = 0.3
real_modulus_foot_pocket = 4e6

foot_pocket_width_at_right_end = 2e-2
foot_pocket_full_width = 10e-2
foot_pocket_width_ramp_length = 15e-2

# Real blade thickness profile along the flexible blade:
# x = 0 is the attachment to the foot pocket, x = 1 is the blade tip.
function real_thickness(x)
    real_thickness_at_attachment = 1.2e-3
    real_thickness_at_tip = 0.7e-3

    # Clamp to use constant material properties for the clamped region and foot pocket.
    x_clamped = clamp(x, 0.0, 1.0)
    return real_thickness_at_attachment +
           x_clamped * (real_thickness_at_tip - real_thickness_at_attachment)
end

# The 2D model represents the blade width in the unmodeled third dimension. The
# foot pocket is narrower than the 19 cm blade, so scale its modulus by the local
# width ratio to preserve the correct out-of-plane integrated stiffness.
function foot_pocket_width(distance_from_right_end)
    ramp_coordinate = clamp(distance_from_right_end / foot_pocket_width_ramp_length,
                            0.0, 1.0)
    width_range = foot_pocket_full_width - foot_pocket_width_at_right_end
    return foot_pocket_width_at_right_end +
           ramp_coordinate * width_range
end

fiber_volume_fraction = 0.6
fiber_density = 1800.0
epoxy_density = 1250.0
real_blade_density = fiber_volume_fraction * fiber_density +
                     (1 - fiber_volume_fraction) * epoxy_density
real_foot_pocket_density = 1000.0
density = real_blade_density

tank_size = (2.0, 1.5)
center = (tank_size[2] / 2, tank_size[2] / 2)
initial_fluid_size = tank_size
initial_velocity = (1.0, 0.0)

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = fin_thickness / (n_particles_y - 1)
fluid_particle_spacing = particle_spacing

smoothing_length_structure = sqrt(2) * particle_spacing
smoothing_length_fluid = 1.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

file = joinpath(examples_dir(), "preprocessing", "data", "hyper_bifins_x.dxf")
geometry = load_geometry(file)

# trixi2vtk(geometry)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=density,
                             grid_offset=center, point_in_geometry_algorithm)
shape_sampled = TrixiParticles.@set shape_sampled.coordinates = Float64.(shape_sampled.coordinates)

# These coordinates are before the final translation to the center position.
foot_pocket_left_end_x = minimum(shape_sampled.coordinates[1, :])
foot_pocket_right_end_x = maximum(shape_sampled.coordinates[1, :])

# The foot pocket is modeled as a rigid structure on the left side,
# and as an elastic structure on the right side.
foot_pocket_rigid_elastic_split_x = -0.27

# Beam and clamped particles
length_clamp = round(Int, 0.15 / particle_spacing) * particle_spacing # m
n_particles_per_dimension = (round(Int, (fin_length + length_clamp) / particle_spacing) + 2,# + n_particles_clamp_x,
                             n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for structures.
# We therefore need to pass `place_on_shell=true`.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (-length_clamp, 0.0), density=density, place_on_shell=true)

# Cut out the beam from the shape to avoid overlapping particles.
foot_pocket = setdiff(shape_sampled, beam)

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
# fluid = setdiff(tank.fluid, structure)

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
    foot_sdf = SignedDistanceField(geometry, particle_spacing;
                                max_signed_distance=4 * particle_spacing,
                                use_for_boundary_packing=true)

    boundary_packing = sample_boundary(foot_sdf; boundary_density=density,
                                    boundary_thickness=4 * particle_spacing)
    boundary_packing = TrixiParticles.@set boundary_packing.coordinates = Float64.(boundary_packing.coordinates)
    boundary_packing = setdiff(boundary_packing, beam)

    background_pressure = 1.0
    smoothing_length_packing = 0.8 * particle_spacing
    foot_packing_system = ParticlePackingSystem(foot_pocket; smoothing_length=smoothing_length_packing,
                                                signed_distance_field=foot_sdf, background_pressure)

    fluid_packing_system = ParticlePackingSystem(boundary_packing; smoothing_length=smoothing_length_packing,
                                                signed_distance_field=foot_sdf, is_boundary=true, background_pressure,
                                                boundary_compress_factor=0.8)

    blade_packing_system = ParticlePackingSystem(beam; smoothing_length=smoothing_length_packing,
                                                fixed_system=true, signed_distance_field=nothing, background_pressure)

    min_corner = minimum(tank.boundary.coordinates, dims=2) .- fluid_particle_spacing / 2
    max_corner = maximum(tank.boundary.coordinates, dims=2) .+ fluid_particle_spacing / 2
    min_corner .-= center
    max_corner .-= center
    cell_list = FullGridCellList(; min_corner, max_corner)
    neighborhood_search = GridNeighborhoodSearch{2}(; cell_list, update_strategy=ParallelUpdate())

    semi_packing = Semidiscretization(foot_packing_system, fluid_packing_system,
                                    blade_packing_system; neighborhood_search)

    ode_packing = semidiscretize(semi_packing, (0.0, 10.0))

    sol_packing = solve(ode_packing, RDPK3SpFSAL35();
                abstol=1e-8,
                save_everystep=false,
                callback=CallbackSet(InfoCallback(interval=50),
                                    #  SolutionSavingCallback(interval=50, prefix="packing_foot"),
                                    UpdateCallback()),
                dtmax=1e-2)

    packed_foot = InitialCondition(sol_packing, foot_packing_system, semi_packing)

    # Move the fin to the center of the tank
    packed_foot.coordinates .+= center
    beam.coordinates .+= center

    # When particles are too close together, keep the ones from `beam`
    # instead of `packed_foot` to ensure that the blade doesn't have holes.
    foot_pocket = setdiff(packed_foot, beam)
    structure = union(beam, foot_pocket)
    fluid = setdiff(tank.fluid, structure)

    # Pack the fluid against the fin and the tank boundary

    pack_window = TrixiParticles.Polygon(stack([
                                                center .+ [-0.4, -0.08],
                                                center .+ [-0.2, -0.08],
                                                center .+ [-0.06, -0.02],
                                                center .+ [0.62, -0.02],
                                                center .+ [0.62, 0.02],
                                                center .+ [0.05, 0.02],
                                                center .+ [-0.1, 0.1],
                                                center .+ [-0.4, 0.1],
                                                center .+ [-0.4, -0.08]
                                            ]))

    # Then, we extract the particles that fall inside this window
    pack_fluid = intersect(fluid, pack_window)
    # and those outside the window.
    fixed_fluid = setdiff(fluid, pack_fluid)
    fixed_union = union(fixed_fluid, structure)

    fluid_packing_system = ParticlePackingSystem(pack_fluid; smoothing_length=smoothing_length_packing,
                                                signed_distance_field=nothing, background_pressure)

    fixed_packing_system = ParticlePackingSystem(fixed_union; smoothing_length=smoothing_length_packing,
                                                fixed_system=true, signed_distance_field=nothing, background_pressure)

    min_corner = minimum(tank.boundary.coordinates, dims=2) .- fluid_particle_spacing / 2
    max_corner = maximum(tank.boundary.coordinates, dims=2) .+ fluid_particle_spacing / 2
    cell_list = FullGridCellList(; min_corner, max_corner)
    neighborhood_search = GridNeighborhoodSearch{2}(; cell_list, update_strategy=ParallelUpdate())

    semi_packing = Semidiscretization(fluid_packing_system, fixed_packing_system;
                                    neighborhood_search)

    ode_packing = semidiscretize(semi_packing, (0.0, 2.0))

    sol_packing = solve(ode_packing, RDPK3SpFSAL35();
                save_everystep=false,
                callback=CallbackSet(InfoCallback(interval=50),
                                     SolutionSavingCallback(interval=50, prefix="packing"),
                                    UpdateCallback()),
                abstol=1e-8,
                dtmax=1e-2)

    fluid = InitialCondition(sol_packing, fluid_packing_system, semi_packing)
    fluid = union(fluid, fixed_fluid)
else
    # When particles are too close together, keep the ones from `beam`
    # instead of `packed_foot` to ensure that the blade doesn't have holes.
    foot_pocket = setdiff(foot_pocket, beam)

    # Move the fin to the center of the tank.
    foot_pocket.coordinates .+= center
    beam.coordinates .+= center

    fluid = setdiff(tank.fluid, union(beam, foot_pocket))
end

# Foot-pocket reference point in the translated tank coordinates used by `structure`.
foot_pocket_rigid_elastic_split_x += center[1]

function is_clamped_structure_particle(coordinates, particle,
                                       foot_pocket_rigid_elastic_split_x)
    x = coordinates[1, particle]
    return x <= foot_pocket_rigid_elastic_split_x
end

blade = beam
structure = union(blade, foot_pocket)

# Make sure that no overlapping particles have been removed. This should've been
# handled by the `setdiff` calls above.
@assert nparticles(structure) == nparticles(foot_pocket) + nparticles(beam)

# Convert particle x-positions to the relative blade coordinate used by `real_thickness`.
# A value of 0 corresponds to the blade attachment, and a value of 1 corresponds to the tip.
function normalized_blade_coordinate(x)
    return (x - center[1]) / fin_length
end

function height_at_x(initial_condition, x)
    distances = abs.(initial_condition.coordinates[1, :] .- x)
    particles = findall(<=(particle_spacing), distances)

    if isempty(particles)
        closest_x = initial_condition.coordinates[1, argmin(distances)]
        particles = findall(particle -> abs(initial_condition.coordinates[1, particle] -
                                            closest_x) <= particle_spacing,
                            1:nparticles(initial_condition))
    end

    y_coordinates = initial_condition.coordinates[2, particles]
    return maximum(y_coordinates) - minimum(y_coordinates)
end

foot_pocket_height_ramp_length = 0.135
foot_pocket_full_height_x = center[1] - foot_pocket_height_ramp_length
foot_pocket_full_height = height_at_x(foot_pocket, foot_pocket_full_height_x)

function foot_pocket_height_for_properties(x)
    distance_from_center = clamp(center[1] - x, 0.0, foot_pocket_height_ramp_length)
    return foot_pocket_full_height * distance_from_center / foot_pocket_height_ramp_length
end

function foot_pocket_width_ratio_for_properties(x)
    distance_from_center = max(center[1] - x, 0.0)
    return foot_pocket_width(distance_from_center) / blade_width
end

function artificial_structure_thickness(x)
    return max(foot_pocket_height_for_properties(x), fin_thickness)
end

function foot_pocket_material_thickness(x)
    blade_thickness = real_thickness(normalized_blade_coordinate(x))
    foot_pocket_height = foot_pocket_height_for_properties(x)
    return max(foot_pocket_height - blade_thickness, 0.0)
end

function density_for_properties(x)
    blade_thickness = real_thickness(normalized_blade_coordinate(x))
    foot_pocket_thickness = foot_pocket_material_thickness(x)
    artificial_thickness = artificial_structure_thickness(x)

    # The 2D model represents the blade width. Scale the foot-pocket contribution
    # by its local out-of-plane width so the represented mass stays correct.
    blade_mass_per_area = real_blade_density * blade_thickness
    foot_pocket_mass_per_area = real_foot_pocket_density *
                                foot_pocket_width_ratio_for_properties(x) *
                                foot_pocket_thickness

    return (blade_mass_per_area + foot_pocket_mass_per_area) / artificial_thickness
end

function modulus_for_properties(x)
    blade_thickness = real_thickness(normalized_blade_coordinate(x))
    foot_pocket_height = foot_pocket_height_for_properties(x)
    artificial_thickness = artificial_structure_thickness(x)

    foot_pocket_modulus = foot_pocket_width_ratio_for_properties(x) *
                          real_modulus_foot_pocket
    foot_pocket_area_moment = max(foot_pocket_height^3 - blade_thickness^3, 0.0) / 12
    flexural_rigidity = real_modulus * blade_thickness^3 / 12 +
                        foot_pocket_modulus * foot_pocket_area_moment

    return 12 * flexural_rigidity / artificial_thickness^3
end

structure.density .= [density_for_properties(structure.coordinates[1, particle])
                      for particle in 1:nparticles(structure)]
structure.mass .= structure.density .* particle_spacing^2
modulus = [modulus_for_properties(structure.coordinates[1, particle])
           for particle in 1:nparticles(structure)]

clamped_structure_particles = findall(particle -> is_clamped_structure_particle(structure.coordinates,
                                                                                particle,
                                                                                foot_pocket_rigid_elastic_split_x),
                                      1:nparticles(structure))

# Movement function (parameters chosen to match video)
frequency = 1.06 # Hz
amplitude = 0.28 # m
rotation_deg = 30 # degrees
rotation_phase_offset = 0.14 # periods
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

viscosity_structure = ArtificialViscosityMonaghan(alpha=1.0)
structure_system = TotalLagrangianSPHSystem(structure; smoothing_kernel, smoothing_length=smoothing_length_structure,
                                        young_modulus=modulus, poisson_ratio,
                                        clamped_particles=clamped_structure_particles,
                                        clamped_particles_motion=boundary_motion,
                                        boundary_model=boundary_model_structure,
                                        velocity_averaging=TrixiParticles.VelocityAveraging(time_constant=5e-4),
                                        viscosity=viscosity_structure,
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

# ==========================================================================================
# ==== Fluid
fluid_density_calculator = ContinuityDensity()
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(fluid; density_calculator=fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length=smoothing_length_fluid,
                                           viscosity=viscosity_fluid,
                                           density_diffusion,
                                           shifting_technique=ParticleShiftingTechnique(sound_speed_factor=0.2, v_max_factor=0.0),
                                           pressure_acceleration=tensile_instability_control,
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
    # open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())
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

semi = Semidiscretization(fluid_system, boundary_system, open_boundary_system, structure_system; neighborhood_search,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
prefix = ""
saving_callback = SolutionSavingCallback(dt=1/120; prefix)

split_cfl = 1.5
# SSPRK104 CFL = 2.5, 15k RHS evaluations
# CarpenterKennedy2N54 CFL = 1.6, 11k RHS evaluations
# RK4 CFL = 1.2, 12k RHS evaluations
# VerletLeapfrog CFL = 0.5, 6.75k RHS evaluations
# VelocityVerlet CFL = 0.5, 6.75k RHS evaluations
# DPRKN4 CFL = 1.7, 9k RHS evaluations

# function tip_velocity(system::TotalLagrangianSPHSystem, data, t)
#     return data.velocity[2254]
# end
# pp_tip = PostprocessCallback(; tip_velocity, interval=1,
#                             filename="$(prefix)_tip_velocity", write_file_interval=10_000)
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
function plane_vtk(system::WeaklyCompressibleSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    resolution = fluid_particle_spacing / 6
    pvd = TrixiParticles.paraview_collection("out/$(prefix)_plane"; append=t > 0)
    interpolate_plane_2d_vtk(min_corner, max_corner, resolution,
                             semi, semi.systems[1], v_ode, u_ode, include_wall_velocity=true,
                             filename="$(prefix)_plane_$(round(Int, t * 1000))", pvd=pvd, t=t)
    TrixiParticles.vtk_save(pvd)
    return nothing
end
interpolate_cb = PostprocessCallback(; plane_vtk, dt=0.01, filename="plane")

mechanical_work_calculator = MechanicalWorkCalculator(semi.systems[4], semi)
thrust_calculator = ThrustCalculator(semi.systems[4], semi, direction=SVector(1.0, 0.0))
calculator_cb = PostprocessCallback(; mechanical_work_calculator, thrust_calculator,
                                    interval=100, filename="$(prefix)_efficiency",
                                    write_file_interval=1000)

callbacks = CallbackSet(info_callback, saving_callback,
                        stepsize_callback, split_integration, pp_cb, interpolate_cb,
                        calculator_cb,
                        UpdateCallback(), SortingCallback(interval=10_000))

dt_fluid = 1.25e-4
sol = solve(ode,
            # RDPK3SpFSAL35(),
            CarpenterKennedy2N54(williamson_condition=false),
            dt=dt_fluid, # This is overwritten by the stepsize callback
            # reltol=1e-5, abstol=1e-7,
            save_everystep=false, callback=callbacks, maxiters=10^8);
