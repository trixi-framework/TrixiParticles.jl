# Flow past a circular cylinder (vortex street), Tafuni et al. (2018).
# Other literature using this validation:
# Vacandio et al. (2013), Marrone et al. (2013), Calhoun (2002), Liu et al. (1998)

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
factor_d = 0.08

const cylinder_diameter = 0.1
particle_spacing = factor_d * cylinder_diameter

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
sound_speed = 10 * prescribed_velocity

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2])

pipe = RectangularTank(particle_spacing, domain_size, boundary_size, fluid_density,
                       n_layers=boundary_layers, velocity=[prescribed_velocity, 0.0],
                       faces=(false, false, true, true))

# Shift pipe walls in negative x-direction for the inflow
pipe.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^(ndims(pipe.fluid) - 1)

cylinder_center = (5 * cylinder_diameter, domain_size[2] / 2)
cylinder = SphereShape(particle_spacing, cylinder_diameter / 2,
                       cylinder_center, fluid_density, sphere_type=RoundSphere())

fluid = setdiff(pipe.fluid, cylinder)

# ==========================================================================================
# ==== Fluid
wcsph = true

h_factor = 4.0
smoothing_length = h_factor * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = prescribed_velocity * cylinder_diameter / reynolds_number

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)
viscosity = ViscosityAdami(nu=kinematic_viscosity)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           density_diffusion=density_diffusion,
                                           smoothing_length, viscosity=viscosity,
                                           buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
function velocity_function2d(pos, t)
    return SVector(prescribed_velocity, 0.0)
end

open_boundary_model = BoundaryModelTafuni()

boundary_type_in = InFlow()
plane_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; plane=plane_in, plane_normal=flow_direction, open_boundary_layers,
                      density=fluid_density, particle_spacing,
                      boundary_type=boundary_type_in)

reference_velocity_in = velocity_function2d
# At the inlet, neither pressure nor density are prescribed; instead,
# these values are extrapolated from the fluid domain
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
outflow = BoundaryZone(; plane=plane_out, plane_normal=(-flow_direction),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       boundary_type=boundary_type_out)

# At the outlet, we allow the flow to exit freely without imposing any boundary conditions
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
circle = SphereShape(particle_spacing, (cylinder_diameter + particle_spacing) / 2,
                     cylinder_center, fluid_density, n_layers=1,
                     sphere_type=RoundSphere())

# Points for pressure interpolation, located at the wall interface
const data_points = reinterpret(reshape, SVector{ndims(circle), eltype(circle)},
                                circle.coordinates)
const center = SVector(cylinder_center)

calculate_lift_force(system, v_ode, u_ode, semi, t) = nothing
function calculate_lift_force(system::TrixiParticles.FluidSystem, v_ode, u_ode, semi, t)
    force = zero(first(data_points))

    for point in data_points
        values = interpolate_points(point, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                    clip_negative_pressure=false)

        # F = ∑ -p_i * A_i * n_i
        force -= values.pressure * particle_spacing .*
                 TrixiParticles.normalize(point - center)
    end

    return 2 * force[2] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

calculate_drag_force(system, v_ode, u_ode, semi, t) = nothing
function calculate_drag_force(system::TrixiParticles.FluidSystem, v_ode, u_ode, semi, t)
    force = zero(first(data_points))

    for point in data_points
        values = interpolate_points(point, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                    clip_negative_pressure=false)

        # F = ∑ -p_i * A_i * n_i
        force -= values.pressure * particle_spacing .*
                 TrixiParticles.normalize(point - center)
    end

    return 2 * force[1] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, open_boundary_in, open_boundary_out,
                          boundary_system_wall, boundary_system_cylinder)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

output_directory = joinpath(validation_dir(),
                            "out_vortex_street_dp_$(factor_d)D_c_$(sound_speed)_h_factor_$(h_factor)_" *
                            TrixiParticles.type2string(smoothing_kernel))

saving_callback = SolutionSavingCallback(; dt=0.02, prefix="", output_directory)

pp_callback = PostprocessCallback(; dt=0.02,
                                  f_l=calculate_lift_force, f_d=calculate_drag_force,
                                  output_directory, filename="resulting_force",
                                  write_csv=true, write_file_interval=10)

extra_callback = nothing

callbacks = CallbackSet(info_callback, UpdateCallback(), saving_callback,
                        ParticleShiftingCallback(), # To obtain a near-uniform particle distribution in the wake
                        pp_callback, extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
