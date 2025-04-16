# Flow past a circular cylinder (vortex street), Tafuni et al. (2018).
# Other literature using this validation:
# Vacandio et al. (2013), Marrone et al. (2013), Calhoun (2002), Liu et al. (1998)

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
const cylinder_diameter = 0.1

factor_D = 0.08

const particle_spacing = factor_D * cylinder_diameter

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 8

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 50.0)

# Boundary geometry and initial fluid particle positions
domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)

flow_direction = [1.0, 0.0]
reynolds_number = 200
const prescribed_velocity = 1.0

const fluid_density = 1000.0

# For this particular example, it is necessary to have a background pressure.
# Otherwise the suction at the outflow is to big and the simulation becomes unstable.
pressure = 0.0

sound_speed = 10 * prescribed_velocity

state_equation = nothing

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2])

pipe = RectangularTank(particle_spacing, domain_size, boundary_size, fluid_density,
                       pressure=pressure, n_layers=boundary_layers,
                       velocity=[prescribed_velocity, 0.0],
                       faces=(false, false, true, true))

# Shift pipe walls in negative x-direction for the inflow
pipe.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^(ndims(pipe.fluid) - 1)

cylinder_center = (5 * cylinder_diameter, domain_size[2] / 2)
cylinder = SphereShape(particle_spacing, cylinder_diameter / 2,
                       cylinder_center, fluid_density, sphere_type=RoundSphere())

zero_v_region_ = RectangularShape(particle_spacing,
                                  (1, 1) .*
                                  round(Int, 3 * cylinder_diameter / particle_spacing),
                                  (5 * cylinder_diameter - 1.5 * cylinder_diameter,
                                   domain_size[2] / 2 - 1.5 * cylinder_diameter),
                                  density=fluid_density)

zero_v_region = setdiff(zero_v_region_, cylinder)
fluid = union(setdiff(pipe.fluid, zero_v_region_), zero_v_region)

# ==========================================================================================
# ==== Fluid
wcsph = true

h_factor = 4.0
smoothing_length = h_factor * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = prescribed_velocity * cylinder_diameter / reynolds_number

# Alternatively the WCSPH scheme can be used
if wcsph
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=7)
    # alpha = 8 * kinematic_viscosity / (smoothing_length * sound_speed)
    # viscosity = ArtificialViscosityMonaghan(; alpha, beta=0.0)

    viscosity = ViscosityAdami(nu=kinematic_viscosity)

    density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               density_diffusion=density_diffusion,
                                               smoothing_length, viscosity=viscosity,
                                               buffer_size=n_buffer_particles)

else
    state_equation = nothing

    viscosity = ViscosityAdami(nu=kinematic_viscosity)

    fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                               smoothing_length,
                                               sound_speed, viscosity=viscosity,
                                               density_calculator=fluid_density_calculator,
                                               buffer_size=n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary

function velocity_function2d(pos, t)
    # Use this for a time-dependent inflow velocity
    # return SVector(0.5prescribed_velocity * sin(2pi * t) + prescribed_velocity, 0)

    return SVector(prescribed_velocity, 0.0)
end

open_boundary_model = BoundaryModelTafuni()

boundary_type_in = InFlow()
plane_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; plane=plane_in, plane_normal=flow_direction, open_boundary_layers,
                      density=fluid_density, particle_spacing,
                      boundary_type=boundary_type_in)

reference_velocity_in = velocity_function2d
reference_pressure_in = nothing
reference_density_in = nothing
open_boundary_in = OpenBoundarySPHSystem(inflow; fluid_system,
                                         boundary_model=open_boundary_model,
                                         buffer_size=n_buffer_particles,
                                         reference_density=reference_density_in,
                                         reference_pressure=reference_pressure_in,
                                         reference_velocity=reference_velocity_in)

boundary_type_out = OutFlow()
plane_out = ([domain_size[1], 0.0], [domain_size[1], domain_size[2]])
outflow = BoundaryZone(; plane=plane_out, plane_normal=-flow_direction,
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       boundary_type=boundary_type_out)

reference_velocity_out = nothing
reference_pressure_out = nothing
reference_density_out = nothing
open_boundary_out = OpenBoundarySPHSystem(outflow; fluid_system,
                                          boundary_model=open_boundary_model,
                                          buffer_size=n_buffer_particles,
                                          reference_density=reference_density_out,
                                          reference_pressure=reference_pressure_out,
                                          reference_velocity=reference_velocity_out)
# ==========================================================================================
# ==== Boundary
boundary_model = BoundaryModelDummyParticles(pipe.boundary.density, pipe.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=nothing, # free-slip
                                             smoothing_kernel, smoothing_length)

boundary_system_wall = BoundarySPHSystem(pipe.boundary, boundary_model)

boundary_model_cylinder = BoundaryModelDummyParticles(cylinder.density, cylinder.mass,
                                                      AdamiPressureExtrapolation(),
                                                      state_equation=state_equation,
                                                      viscosity=viscosity,
                                                      smoothing_kernel, smoothing_length)

boundary_system_cylinder = BoundarySPHSystem(cylinder, boundary_model_cylinder)

# ==========================================================================================
# ==== Postprocessing
circle = SphereShape(particle_spacing, cylinder_diameter / 2,
                     cylinder_center, fluid_density, n_layers=1,
                     sphere_type=RoundSphere())

const data_points = reinterpret(reshape, SVector{ndims(circle), eltype(circle)},
                                circle.coordinates)
const center = SVector(cylinder_center)

calculate_lift_force(system, v_ode, u_ode, semi, t) = nothing
function calculate_lift_force(system::TrixiParticles.FluidSystem, v_ode, u_ode, semi, t)
    force = zero(first(data_points))

    for point in data_points
        values = interpolate_point(point, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                   clip_negative_pressure=false)

        # F = ∑ -p_i * A_i * n_i
        force -= values.pressure * TrixiParticles.normalize(point - center) *
                 particle_spacing
    end

    return 2 * force[2] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

calculate_drag_force(system, v_ode, u_ode, semi, t) = nothing
function calculate_drag_force(system::TrixiParticles.FluidSystem, v_ode, u_ode, semi, t)
    force = zero(first(data_points))

    for point in data_points
        values = interpolate_point(point, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                   clip_negative_pressure=false)

        # F = ∑ -p_i * A_i * n_i
        force -= values.pressure * TrixiParticles.normalize(point - SVector(center)) *
                 particle_spacing
    end

    return 2 * force[1] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, open_boundary_in, open_boundary_out,
                          boundary_system_wall, boundary_system_cylinder)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

# saving_callback = SolutionSavingCallback(dt=0.02, prefix="",
#                                          output_directory="out_vortex_street_dp_$(factor_D)D")

pp_callback = PostprocessCallback(; dt=0.02,
                                  f_l=calculate_lift_force,
                                  f_d=calculate_drag_force,
                                  output_directory="out_vortex_street_dp_$(factor_D)D_h_factor_$(h_factor)_$(smoothing_kernel)_c_$sound_speed",
                                  filename="resulting_force",
                                  write_csv=true, write_file_interval=10)

extra_callback = nothing

callbacks = CallbackSet(info_callback,  UpdateCallback(), # saving_callback,
                        ParticleShiftingCallback(), pp_callback, extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
