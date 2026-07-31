using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSymplecticRK

include("fin_2d/setup.jl")

function convert_ic(ic, T)
    return InitialCondition{ndims(ic)}(ic.coordinates, ic.velocity, ic.mass, ic.density,
                                      ic.pressure, T(ic.particle_spacing))
end

# ==========================================================================================
# ==== Resolution
n_particles_y = 4

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 3.0)

fin_length = 0.522
fin_thickness = 30e-3
blade_width = 19e-2
real_modulus = 40e9
poisson_ratio = 0.3
real_modulus_foot_pocket = 5e5

foot_pocket_width_at_right_end = 2e-2
foot_pocket_full_width = 10e-2
foot_pocket_width_ramp_length = 15e-2

# Real blade thickness profile along the flexible blade:
# x = 0 is the attachment to the foot pocket, x = 1 is the blade tip.
function real_thickness(x)
    real_thickness_at_attachment = 1.2e-3
    real_thickness_at_tip = 0.7e-3

    # `p = 1` is a linear profile.
    p = 1

    # Clamp to use constant material properties for the clamped region and foot pocket.
    x_clamped = clamp(x, 0.0, 1.0)
    return real_thickness_at_tip +
           (1 - x_clamped)^p * (real_thickness_at_attachment - real_thickness_at_tip)
end

# Real blade width profile along the flexible blade, where `x` is the distance
# from the blade attachment in meters.
function real_blade_width(x)
    if x > 0.12
        return blade_width
    end
    width = -1.199 * x^2 + 0.346 * x + 0.167
    return clamp(width, 7.5e-2, blade_width)
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

function foot_pocket_height_top(x_normalized)
    # Linear approximation of the top edge of the foot pocket.
    # This is only used to limit the material interface blending width, so it doesn't
    # need to be exact, and it can be clamped once it's larger than the blending width.
    return clamp(-0.086 * x_normalized / 0.135, 0.0, 0.08)
end

function foot_pocket_height_bottom(x_normalized)
    # Cubic polynomial fitted to the bottom edge of the foot pocket.
    # This is only used to limit the material interface blending width, so it doesn't
    # need to be exact, and it can be clamped once it's larger than the blending width.
    x = clamp(x_normalized, -0.3, -0.1)
    return clamp(2.09 * x^3 + 1.25 * x^2 + 0.135 * x + 0.003, 0.0, 0.015)
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

simulate_foot_pocket = true

# When the foot pocket is simulated, extend the blade into the foot pocket to guarantee
# a good particle distribution. Without a foot pocket, clamp 1cm of the blade.
length_clamp = simulate_foot_pocket ? 0.3 : 0.01
length_clamp = round(Int, length_clamp / particle_spacing) * particle_spacing # m

n_particles_per_dimension = (round(Int, (fin_length + length_clamp) / particle_spacing) + 1,
                             n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for structures.
# We therefore need to pass `place_on_shell=true`.
blade = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (-length_clamp, -fin_thickness / 2), density=density,
                        place_on_shell=true)

# The foot pocket is modeled as a rigid structure on the left side,
# and as an elastic structure on the right side.
rigid_elastic_split_x = -0.27

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
    foot_pocket, fluid = sample_and_pack(particle_spacing, center, blade, tank.fluid)
    fin = union(blade, foot_pocket)
else
    foot_pocket = sample_foot_pocket(particle_spacing, center, blade)

    # Move the fin to the center of the tank. This is done automatically
    # in `sample_and_pack`.
    foot_pocket.coordinates .+= center
    blade.coordinates .+= center

    fin = union(blade, foot_pocket)
    fluid = setdiff(tank.fluid, fin)
end

# Foot-pocket reference point in the translated tank coordinates used by `structure`.
rigid_elastic_split_x += center[1]

function is_clamped_structure_particle(coordinates, particle,
                                       rigid_elastic_split_x)
    x = coordinates[1, particle]
    return x <= rigid_elastic_split_x
end

structure = union(blade, foot_pocket)

# Make sure that no overlapping particles have been removed. This should've been
# handled by the `setdiff` calls above.
@assert nparticles(structure) == nparticles(foot_pocket) + nparticles(blade)

# Convert particle x-positions to the relative blade coordinate used by `real_thickness`.
# A value of 0 corresponds to the blade attachment, and a value of 1 corresponds to the tip.
function normalized_blade_coordinate(x)
    return (x - center[1]) / fin_length
end

function foot_pocket_width_ratio_for_properties(x)
    distance_from_center = max(center[1] - x, 0.0)
    return foot_pocket_width(distance_from_center) / blade_width
end

# Blend the material interface around the edge of the artificially thick blade.
# The discontinuity is half a particle spacing beyond the outer blade particles.
material_discontinuity_distance = fin_thickness / 2 + particle_spacing / 2

@inline function local_material_blend_widths(foot_pocket_height)
    material_blend_outer_width = fin_thickness * 5 / 6
    available_outer_height = max(foot_pocket_height - material_discontinuity_distance, 0.0)
    height_scale = min(available_outer_height / material_blend_outer_width, 1.0)

    # Reduce both sides by the same factor when the surrounding pocket is too thin.
    inner_width = height_scale * fin_thickness / 6
    outer_width = height_scale * material_blend_outer_width

    # A sub-particle transition cannot be resolved. Treat it as sharp instead.
    if inner_width + outer_width < fluid_particle_spacing
        return 0.0, 0.0
    end

    return inner_width, outer_width
end

@inline function log_linear_blend(left_value, right_value, alpha)
    if !(left_value > 0 && right_value > 0)
        throw(DomainError((left_value, right_value),
                          "log-linear blending requires positive endpoint values"))
    end

    alpha <= 0 && return left_value
    alpha >= 1 && return right_value

    return exp(log(left_value) +
               alpha * (log(right_value) - log(left_value)))
end

@inline function blade_blend_alpha(x, y, blade_modulus, foot_pocket_modulus)
    # There is no material interface along the free part of the blade.
    x >= center[1] && return 0.0

    blade_center_y = center[2]

    distance_from_blade_center = abs(y - blade_center_y)
    foot_pocket_height = if y >= blade_center_y
        foot_pocket_height_top(x - center[1])
    else
        foot_pocket_height_bottom(x - center[1])
    end
    inner_width, outer_width = local_material_blend_widths(foot_pocket_height)

    if iszero(inner_width + outer_width)
        return distance_from_blade_center <= material_discontinuity_distance ? 0.0 : 1.0
    end

    inner_edge = material_discontinuity_distance - inner_width
    outer_edge = material_discontinuity_distance + outer_width
    distance_from_blade_center <= inner_edge && return 0.0
    distance_from_blade_center >= outer_edge && return 1.0

    return clamp((distance_from_blade_center - inner_edge) / (outer_edge - inner_edge),
                 0, 1)
end

function material_property_endpoints(x)
    normalized_x = normalized_blade_coordinate(x)
    real_blade_thickness = real_thickness(normalized_x)
    foot_pocket_width_ratio = foot_pocket_width_ratio_for_properties(x)

    foot_pocket_density = foot_pocket_width_ratio * real_foot_pocket_density
    blade_density = real_blade_density * real_blade_thickness / fin_thickness

    real_width = real_blade_width(clamp(normalized_x, 0.0, 1.0) * fin_length)
    flexural_rigidity = real_modulus * real_width * real_blade_thickness^3 / 12
    blade_modulus = flexural_rigidity * 12 / (blade_width * fin_thickness^3)
    foot_pocket_modulus = foot_pocket_width_ratio * real_modulus_foot_pocket

    return (; blade_density, foot_pocket_density, blade_modulus,
            foot_pocket_modulus)
end

function density_for_properties(x, y)
    properties = material_property_endpoints(x)
    alpha = blade_blend_alpha(x, y, properties.blade_modulus,
                              properties.foot_pocket_modulus)

    return log_linear_blend(properties.blade_density, properties.foot_pocket_density, alpha)
end

function modulus_for_properties(x, y)
    properties = material_property_endpoints(x)
    alpha = blade_blend_alpha(x, y, properties.blade_modulus,
                              properties.foot_pocket_modulus)

    return log_linear_blend(properties.blade_modulus, properties.foot_pocket_modulus, alpha)
end

const FIN_MOTION_FREQUENCY = 1.06
const FIN_MOTION_PERIOD_START = 1.0
const FIN_MOTION_REFERENCE = SVector(center[1], center[2] + fin_thickness / 2)
const FIN_TRANSLATION_X_COEFFICIENTS = (-12.806917764769953, -3.0303457592946477, -2.3619440315595286, -6.596342617780676, 10.147967943261595, 1.6701714018007443, -0.753248176125403, 1.1049047920757982, -2.309571815723685)
const FIN_TRANSLATION_Y_COEFFICIENTS = (8.238789990145717, 36.492667336354685, 233.1179370123477, 2.748787827252771, -0.542272634784335, 6.639630128273474, -6.0841959049116765, 0.48789188369953673, -0.14154297148482692)
const FIN_ROTATION_COEFFICIENTS = (0.9821743002411218, -43.87937992072254, 13.295138772155541, 0.8057808770021837, 0.14656663179778703, 3.0936007063613857, 0.15219421833610516, -0.26097594464357143, 0.06084159719766363)

@inline function spectral_value(t, coefficients)
    theta = 2pi * FIN_MOTION_FREQUENCY *
            (t - FIN_MOTION_PERIOD_START)
    value = coefficients[1]

    @inbounds for harmonic in 1:((length(coefficients) - 1) ÷ 2)
        sine, cosine = sincos(harmonic * theta)
        value += coefficients[2harmonic] * cosine +
                 coefficients[2harmonic + 1] * sine
    end

    return value
end

@inline function fitted_movement(x, t)
    # Smooth startup matching the previous 0.5 s ramp.
    tau = clamp(t / 0.5, 0, 1)
    ramp = tau^3 * (10 + tau * (-15 + 6tau))

    translation = 1e-3 * SVector(
        spectral_value(t, FIN_TRANSLATION_X_COEFFICIENTS),
        spectral_value(t, FIN_TRANSLATION_Y_COEFFICIENTS),
    )

    angle = deg2rad(spectral_value(t, FIN_ROTATION_COEFFICIENTS))

    sine, cosine = sincos(angle)
    relative_position = x - FIN_MOTION_REFERENCE
    rotated_position = SVector(
        cosine * relative_position[1] - sine * relative_position[2],
        sine * relative_position[1] + cosine * relative_position[2],
    )
    target_position = FIN_MOTION_REFERENCE + rotated_position + translation

    # Ramp the complete displacement, as done by `OscillatingMotion2D`.
    return x + ramp * (target_position - x)
end

if simulate_foot_pocket
    # Movement function (parameters chosen to match video)
    frequency = 1.06 # Hz
    amplitude = 0.24 # m
    rotation_deg = 22 # degrees
    rotation_phase_offset = 0.18 # periods
    rotation_center = center
    rotation_angle = rotation_deg * pi / 180
    boundary_motion = OscillatingMotion2D(; frequency,
                                          translation_vector=SVector(0.0, amplitude),
                                          rotation_angle, rotation_center,
                                          rotation_phase_offset, ramp_up_tspan=(0.0, 0.5))

else
    structure = blade
    rigid_elastic_split_x = center[1]

    boundary_motion = PrescribedMotion(fitted_movement, Returns(true))
end

structure.density .= [density_for_properties(structure.coordinates[1, particle],
                                             structure.coordinates[2, particle])
                      for particle in 1:nparticles(structure)]
structure.mass .= structure.density .* particle_spacing^2
modulus = [modulus_for_properties(structure.coordinates[1, particle],
                                  structure.coordinates[2, particle])
           for particle in 1:nparticles(structure)]

clamped_structure_particles = findall(particle -> is_clamped_structure_particle(structure.coordinates,
                                                                                particle,
                                                                                rigid_elastic_split_x),
                                      1:nparticles(structure))

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
solution_prefix = ""
saving_callback = SolutionSavingCallback(dt=1/120; prefix=solution_prefix)

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
#                             filename="$(solution_prefix)_tip_velocity", write_file_interval=10_000)
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
                            filename="$(solution_prefix)_total_volume", write_file_interval=50)

efficiency_interval = 100
mechanical_work_calculator = MechanicalWorkCalculator(semi.systems[4], semi)
thrust_calculator = ThrustCalculator(semi.systems[4], semi, direction=SVector(1.0, 0.0))
calculator_cb = PostprocessCallback(; mechanical_work_calculator, thrust_calculator,
                                    interval=efficiency_interval, write_file_interval=10,
                                    filename="$(solution_prefix)_efficiency")

# Reconstruct the motion of the blade centerline at its attachment (`x = 0` in blade
# coordinates) from the surrounding SPH particles. The displacement and deformation
# gradient are interpolated in the initial configuration with volume-weighted kernel
# values and Shepard normalization. The rotation is the rotational part of the
# interpolated deformation gradient.
const BLADE_ATTACHMENT_REFERENCE = FIN_MOTION_REFERENCE
blade_motion_system = semi.systems[4]
blade_motion_search_radius = TrixiParticles.compact_support(blade_motion_system,
                                                            blade_motion_system)

# The reference configuration is fixed, so determine the kernel support only once.
const BLADE_MOTION_PARTICLES = findall(1:nparticles(blade_motion_system)) do particle
    initial_position = TrixiParticles.initial_coords(blade_motion_system, particle)
    distance2 = sum(abs2, initial_position - BLADE_ATTACHMENT_REFERENCE)
    return distance2 <= blade_motion_search_radius^2
end
@assert !isempty(BLADE_MOTION_PARTICLES)

const BLADE_MOTION_WEIGHTS = map(BLADE_MOTION_PARTICLES) do particle
    initial_position = TrixiParticles.initial_coords(blade_motion_system, particle)
    distance = sqrt(sum(abs2, initial_position - BLADE_ATTACHMENT_REFERENCE))
    kernel_weight = TrixiParticles.kernel(blade_motion_system.smoothing_kernel, distance,
                                          blade_motion_system.smoothing_length)
    volume = blade_motion_system.mass[particle] /
             blade_motion_system.material_density[particle]
    return volume * kernel_weight
end
const BLADE_MOTION_WEIGHT_SUM = sum(BLADE_MOTION_WEIGHTS)
@assert BLADE_MOTION_WEIGHT_SUM > eps(BLADE_MOTION_WEIGHT_SUM)

function blade_attachment_motion(system)
    displacement = zero(BLADE_ATTACHMENT_REFERENCE)
    deformation_grad = zero(TrixiParticles.deformation_gradient(system,
                                                                 first(BLADE_MOTION_PARTICLES)))

    @inbounds for i in eachindex(BLADE_MOTION_PARTICLES)
        particle = BLADE_MOTION_PARTICLES[i]
        weight = BLADE_MOTION_WEIGHTS[i]

        displacement += weight * (TrixiParticles.current_coords(system, particle) -
                                  TrixiParticles.initial_coords(system, particle))
        deformation_grad += weight *
                            TrixiParticles.deformation_gradient(system, particle)
    end

    displacement /= BLADE_MOTION_WEIGHT_SUM
    deformation_grad /= BLADE_MOTION_WEIGHT_SUM

    # In 2D, this is the angle of the proper orthogonal factor in the polar
    # decomposition of the deformation gradient.
    rotation = atan(deformation_grad[2, 1] - deformation_grad[1, 2],
                    deformation_grad[1, 1] + deformation_grad[2, 2])

    return displacement, rotation
end

function blade_motion(system::TotalLagrangianSPHSystem, data, t)
    translation, rotation = blade_attachment_motion(system)
    return translation, rotation
end

blade_motion(system, data, t) = nothing

blade_motion_cb = PostprocessCallback(; blade_motion, dt=1 / 120,
                                      write_file_interval=10,
                                      filename="$(solution_prefix)_blade_motion")

callbacks = CallbackSet(info_callback, saving_callback,
                        stepsize_callback, split_integration, pp_cb,
                        calculator_cb, blade_motion_cb,
                        UpdateCallback(), SortingCallback(interval=10_000))

dt_fluid = 1.25e-4
sol = solve(ode,
            # RDPK3SpFSAL35(),
            CarpenterKennedy2N54(williamson_condition=false),
            dt=dt_fluid, # This is overwritten by the stepsize callback
            # reltol=1e-5, abstol=1e-7,
            save_everystep=false, callback=callbacks, maxiters=10^8);
