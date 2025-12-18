# This file computes the pressure sensor data of the dam break setup described in
#
# J. J. De Courcy, T. C. S. Rendall, L. Constantin, B. Titurus, J. E. Cooper.
# "Incompressible δ-SPH via artificial compressibility".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 420 (2024),
# https://doi.org/10.1016/j.cma.2023.116700

using TrixiParticles
using TrixiParticles.JSON
using CUDA


function max_x_coord(system, data, t)
    return maximum(view(data.coordinates, 1, :))
end

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t, system, semi) end

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t,
                               system::TrixiParticles.AbstractFluidSystem, semi)
    n_interpolation_points = 10
    interpolated_values = interpolate_line(coord_top, coord_bottom,
                                           n_interpolation_points, semi, system, v_ode,
                                           u_ode,
                                           #    smoothing_length=2.0 *
                                           #                     TrixiParticles.initial_smoothing_length(system),
                                           clip_negative_pressure=true, cut_off_bnd=false)
    return sum(map(x -> isnan(x) ? 0.0 : x, interpolated_values.pressure)) /
           n_interpolation_points
end
# ==========================================================================================

# When using data center CPUs with large numbers of cores, especially on multi-socket
# systems with multiple NUMA nodes, pinning threads to cores can significantly
# improve performance, even for low resolutions.
# using ThreadPinning
# pinthreads(:numa)

# `resolution` in this case is set relative to `H`, the initial height of the fluid.
# Use 40, 80 or 400 for validation.
# Note: 400 takes about 30 minutes on a large data center CPU (much longer with serial update)
particles_per_height = 600

min_corner = (-1.5, -1.5)
max_corner = (6.5, 5.0)
cell_list = FullGridCellList(; min_corner, max_corner, max_points_per_cell=30)

neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                update_strategy=ParallelUpdate())

# ==========================================================================================
# ==== WCSPH simulation
# trixi_include_changeprecision(Float32, @__MODULE__,
#                               joinpath(validation_dir(), "dam_break_2d",
#                                        "setup_marrone_2011.jl"),
#                               parallelization_backend=CUDABackend())




# 2D dam break validation setup based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016
#
# and
#
# J. J. De Courcy, T. C. S. Rendall, L. Constantin, B. Titurus, J. E. Cooper.
# "Incompressible δ-SPH via artificial compressibility".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 420 (2024),
# https://doi.org/10.1016/j.cma.2023.116700


H = 0.6
fluid_particle_spacing = H / particles_per_height
W = 2 * H


# Import variables from the example file
alpha = 0.01
viscosity_fluid = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
shifting_technique = nothing
# trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
#               alpha=0.01,
#               sol=nothing,
#               ode=nothing,
#               shifting_technique=shifting_technique,
#               viscosity_fluid=viscosity_fluid,
#               fluid_particle_spacing=particle_spacing)


boundary_layers = 4

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
# Boundary geometry and initial fluid particle positions
initial_fluid_size = (W, H)
# tank_size = (floor(5.366 * H / boundary_particle_spacing) * boundary_particle_spacing,
#              2.0+10*fluid_particle_spacing)
tank_size = (floor(5.366 * H / fluid_particle_spacing) * fluid_particle_spacing,
             2.0)

fluid_density = 1000.0
sound_speed = 20 * sqrt(gravity * H)
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

# The density diffusion model by Molteni and Colagrossi shows unphysical effects at the
# free surface in long-running simulations, but is significantly faster than the model
# by Antuono. This simulation is short enough to use the faster model.
# density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
density_diffusion = DensityDiffusionAntuono(tank.fluid, delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity_fluid,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity), correction=nothing,
                                           surface_tension=nothing,
                                           reference_particle_spacing=fluid_particle_spacing,
                                           shifting_technique=nothing)

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
viscosity_wall = nothing
# For a no-slip boundary condition, define a wall viscosity:
# viscosity_wall = viscosity_fluid
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             correction=nothing,
                                             reference_particle_spacing=fluid_particle_spacing,
                                             viscosity=viscosity_wall)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model,
                                     adhesion_coefficient=0.0)

# ==========================================================================================
# ==== Simulation
# `nothing` will automatically choose the best update strategy. This is only to be able
# to change this with `trixi_include`.
extra_system = nothing
extra_system2 = nothing
semi = Semidiscretization(fluid_system, boundary_system, extra_system, extra_system2,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

solution_prefix = "validation_" * method * "_" * formatted_string
saving_callback = SolutionSavingCallback(dt=0.01, prefix=solution_prefix)

# This can be overwritten with `trixi_include`
extra_callback = nothing
extra_callback2 = nothing

use_reinit = false
density_reinit_cb = use_reinit ?
                    DensityReinitializationCallback(semi.systems[1], interval=10) :
                    nothing
stepsize_callback = StepsizeCallback(cfl=0.9)


tspan = (0.0, 7 / sqrt(9.81 / 0.6)) # This is used by De Courcy et al. (2024)

fluid_density = 1000.0

# Sensor positions used by De Courcy et al. (2024)
sensor_size = 0.0084
P1_y_top = (6 + 4.2) / 600 * H
P1_y_bottom = P1_y_top - sensor_size
P2_y_top = (30 + 4.2) / 600 * H
P2_y_bottom = P2_y_top - sensor_size
P3_y_top = (60 + 4.2) / 600 * H
P3_y_bottom = P3_y_top - sensor_size
P4_y_top = (160 + 4.2) / 600 * H
P4_y_bottom = P4_y_top - sensor_size

tank_right_wall_x = tank_size[1]

pressure_P1 = (system, dv_ode, du_ode, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P1_y_top],
                                           [tank_right_wall_x, P1_y_bottom],
                                           v_ode, u_ode, t, system, semi)
pressure_P2 = (system, dv_ode, du_ode, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P2_y_top],
                                           [tank_right_wall_x, P2_y_bottom],
                                           v_ode, u_ode, t, system, semi)
pressure_P3 = (system, dv_ode, du_ode, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P3_y_top],
                                           [tank_right_wall_x, P3_y_bottom],
                                           v_ode, u_ode, t, system, semi)
pressure_P4 = (system, dv_ode, du_ode, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P4_y_top],
                                           [tank_right_wall_x, P4_y_bottom],
                                           v_ode, u_ode, t, system, semi)

method = "wcsph"


extra_string = ""
formatted_string = string(particles_per_height) * extra_string

postprocessing_cb = PostprocessCallback(; dt=0.01 / sqrt(gravity / H),
                                        output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, exclude_boundary=false,
                                        max_x_coord,
                                        pressure_P1, pressure_P2,
                                        pressure_P3, pressure_P4)

# Disable loop flipping to produce consistent results over different thread numbers
boundary_density_calculator = AdamiPressureExtrapolation(allow_loop_flipping=false)

# Save at certain timepoints which allows comparison to the results of Marrone et al.,
# i.e. t(g/H)^(1/2) = (1.5, 2.36, 3.0, 5.7, 6.45).
# Note that the images in Marrone et al. are obtained with `particles_per_height = 320`.
saving_paper = SolutionSavingCallback(save_times=[0.0, 1.5, 2.36, 3.0, 5.7, 6.45] ./
                                                 sqrt(gravity / H), prefix="marrone_times")

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback, postprocessing_cb,
                        saving_paper, density_reinit_cb)


# time_integration_scheme = CarpenterKennedy2N54(williamson_condition=false)
# sol = solve(ode, time_integration_scheme,
#             dt=1.0, # This is overwritten by the stepsize callback
#             save_everystep=false, callback=callbacks);
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-6, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-4, # Limit stepsize to prevent crashing
            maxiters=1e7,
            save_everystep=false, callback=callbacks);
