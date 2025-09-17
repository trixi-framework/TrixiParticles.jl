# ==========================================================================================
# 2D Poiseuille Flow Simulation (Weakly Compressible SPH)
#
# Based on:
#   Zhan, X., et al. "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics"
#   Physics of Fluids, Volume 37
#   https://doi.org/10.1063/5.0254575
#
# This example sets up a 2D Poiseuille flow simulation in a rectangular channel
# including open boundary conditions and
# transient analytical solution for comparison and error analysis.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using Plots

# ==========================================================================================
# ==== Resolution
const wall_distance = 0.001
const flow_length = 0.004

particle_spacing = wall_distance / 50

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 10

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)
wcsph = true

domain_size = (flow_length, wall_distance)

open_boundary_size = (open_boundary_layers * particle_spacing, domain_size[2])

fluid_density = 1000.0
reynolds_number = 50
const pressure_drop = 0.1
const dynamic_viscosity = sqrt(fluid_density * wall_distance^3 * pressure_drop /
                               (8 * flow_length * reynolds_number))

v_max = wall_distance^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

sound_speed_factor = 100
sound_speed = sound_speed_factor * v_max

flow_direction = (1.0, 0.0)

function poiseuille_velocity(y, t)

    # Base profile (stationary part)
    base_profile = (pressure_drop / (2 * dynamic_viscosity * flow_length)) * y *
                   (y - wall_distance)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in 0:10  # Limit to 10 terms for convergence
        coefficient = (4 * pressure_drop * wall_distance^2) /
                      (dynamic_viscosity * flow_length * π^3 * (2 * n + 1)^3)

        sine_term = sin(π * y * (2 * n + 1) / wall_distance)

        exp_term = exp(-((2 * n + 1)^2 * π^2 * dynamic_viscosity * t) /
                       (fluid_density * wall_distance^2))

        transient_sum += coefficient * sine_term * exp_term
    end

    # Total velocity
    v_x = base_profile + transient_sum

    return v_x
end

# For the inflow velocity
function velocity_profile(pos, t)
    y = pos[2]  # y-coordinate
    v_x = poiseuille_velocity(y, t)

    return (-v_x, 0.0)  # (v_x, v_y)
end

pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density,
                       velocity=(pos) -> velocity_profile(pos, 0),
                       pressure=(pos) -> 0.2 + (0.1 - 0.2) * (pos[1] / flow_length),
                       n_layers=boundary_layers, faces=(false, false, true, true))

inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers,
                        velocity=(pos) -> velocity_profile(pos, 0), pressure=0.2,
                        min_coordinates=(0.0, 0.0),
                        faces=(false, false, true, true))

outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         velocity=(pos) -> velocity_profile(pos, 0), pressure=0.1,
                         min_coordinates=(pipe.fluid_size[1] - open_boundary_size[1], 0.0),
                         faces=(false, false, true, true))

fluid = setdiff(pipe.fluid, inlet.fluid, outlet.fluid)

n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^2

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = dynamic_viscosity / fluid_density

viscosity = ViscosityAdami(nu=kinematic_viscosity)

background_pressure = 7 * sound_speed_factor / 10 * fluid_density * v_max^2
shifting_technique = TransportVelocityAdami(; background_pressure)

if wcsph
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=1)
    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               buffer_size=n_buffer_particles,
                                               shifting_technique=shifting_technique,
                                               density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                               smoothing_length, viscosity=viscosity)
else
    state_equation = nothing

    fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                               smoothing_length,
                                               sound_speed, viscosity=viscosity,
                                               density_calculator=fluid_density_calculator,
                                               shifting_technique=shifting_technique,
                                               buffer_size=n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelDynamicalPressureZhang()

boundary_type_in = BidirectionalFlow()
plane_in = ([open_boundary_size[1], 0.0], [open_boundary_size[1], pipe.fluid_size[2]])
reference_velocity_in = nothing
reference_pressure_in = 0.2
inflow = BoundaryZone(; plane=plane_in, plane_normal=flow_direction, open_boundary_layers,
                      density=fluid_density, particle_spacing, rest_pressure=0.2,
                      reference_velocity=reference_velocity_in,
                      reference_pressure=reference_pressure_in,
                      initial_condition=inlet.fluid, boundary_type=boundary_type_in)

boundary_type_out = BidirectionalFlow()
plane_out = ([pipe.fluid_size[1] - open_boundary_size[1], 0.0],
             [pipe.fluid_size[1] - open_boundary_size[1], pipe.fluid_size[2]])
reference_velocity_out = nothing
reference_pressure_out = 0.1
outflow = BoundaryZone(; plane=plane_out, plane_normal=(.-(flow_direction)),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       rest_pressure=0.1,
                       reference_velocity=reference_velocity_out,
                       reference_pressure=reference_pressure_out,
                       initial_condition=outlet.fluid, boundary_type=boundary_type_out)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
wall = union(pipe.boundary)

boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=viscosity,
                                             smoothing_kernel, smoothing_length)

boundary_system = WallBoundarySystem(wall, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(wall.coordinates .- 2 * particle_spacing, dims=2)
max_corner = maximum(wall.coordinates .+ 2 * particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{2}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary,
                          boundary_system, neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

v_x_interpolated(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function v_x_interpolated(system::TrixiParticles.AbstractFluidSystem,
                          dv_ode, du_ode, v_ode, u_ode, semi, t)
    start_point = [flow_length / 2, 0.0]
    end_point = [flow_length / 2, wall_distance]

    values = interpolate_line(start_point, end_point, 500, semi, system, v_ode, u_ode;
                              cut_off_bnd=false, clip_negative_pressure=false)

    return values.velocity[1, :]
end

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="", output_directory="out")

pp_callback = PostprocessCallback(; dt=0.02, output_directory="out",
                                  v_x=v_x_interpolated, filename="result",
                                  write_csv=true, write_file_interval=1)
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(),
                        pp_callback, extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
