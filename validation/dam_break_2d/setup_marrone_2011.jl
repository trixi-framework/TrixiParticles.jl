# 2D dam break validation setup based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

include("sensors.jl")

using TrixiParticles
using JSON

use_edac = true # Set to false to use WCSPH

gravity = 9.81
H = 0.6
W = 2 * H
tspan = (0.0, 8.0 / sqrt(gravity / H))

# `particle_spacing` in this case is set relative to `H`, the initial height of the fluid.
# Use H / 80, H / 320 for validation.
# Note: H / 320 takes a few hours!
particles_per_height = 320
particle_spacing = H / particles_per_height
# For reference:
# H/320 is 0.001875m
# H/80 is 0.0075m
# H/40 is 0.015m
smoothing_length = 1.32 * particle_spacing
smoothing_kernel = GaussianKernel{2}()

fluid_density = 1000.0
sound_speed = 40 * sqrt(gravity * H)

boundary_layers = 4
spacing_ratio = 1
boundary_particle_spacing = particle_spacing / spacing_ratio

initial_fluid_size = (W, H)
tank_size = (floor(5.366 * H / boundary_particle_spacing) * boundary_particle_spacing, 4.0)

# The top of the sensors is at the position where the middle of the sensors is in the experiment.
#
# > Note, the probe position in the numerical setup does not exactly match the position in the experiment, but
# > Greco [36] showed that this shift gives better agreement and reported several uncertainties in the
# > measurements motivating this adjustment.
# S. Adami, X. Y. Hu, N. A. Adams.
# "A generalized wall boundary condition for smoothed particle hydrodynamics".
# In: Journal of Computational Physics 231, 21 (2012), pages 7057--7075.
# https://doi.org/10.1016/J.JCP.2012.05.005
sensor_size = 0.09
P1_y_top = 160 / 600 * H
P1_y_bottom = P1_y_top - sensor_size
P2_y_top = (160 + 424) / 600 * H
P2_y_bottom = P2_y_top - sensor_size
P3_y_top = (160 + 424 + 416) / 600 * H
P3_y_bottom = P3_y_top - sensor_size

tank_right_wall_x = floor(5.366 * H / particle_spacing) * particle_spacing -
                    0.5 * particle_spacing

interpolated_pressure_P1 = (system, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P1_y_top],
                                           [tank_right_wall_x, P1_y_bottom],
                                           v_ode, u_ode, t, system, semi)
interpolated_pressure_P2 = (system, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P2_y_top],
                                           [tank_right_wall_x, P2_y_bottom],
                                           v_ode, u_ode, t, system, semi)
interpolated_pressure_P3 = (system, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P3_y_top],
                                           [tank_right_wall_x, P3_y_bottom],
                                           v_ode, u_ode, t, system, semi)

particle_pressure_P1 = (system, v_ode, u_ode, semi,
               t) -> pressure_probe([tank_right_wall_x, P1_y_top],
                                           [tank_right_wall_x, P1_y_bottom],
                                           v_ode, u_ode, t, system, semi)
particle_pressure_P2 = (system, v_ode, u_ode, semi,
               t) -> pressure_probe([tank_right_wall_x, P2_y_top],
                                           [tank_right_wall_x, P2_y_bottom],
                                           v_ode, u_ode, t, system, semi)
particle_pressure_P3 = (system, v_ode, u_ode, semi,
               t) -> pressure_probe([tank_right_wall_x, P3_y_top],
                                           [tank_right_wall_x, P3_y_bottom],
                                           v_ode, u_ode, t, system, semi)

if use_edac
    method = "edac"
    state_equation = nothing

    tank_edac = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                            n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                            acceleration=(0.0, -gravity), state_equation=nothing)

    alpha = 0.02
    viscosity_edac = ViscosityAdami(nu=alpha * smoothing_length * sound_speed / 8)
    fluid_system = EntropicallyDampedSPHSystem(tank_edac.fluid, smoothing_kernel,
                                                    smoothing_length,
                                                    sound_speed, viscosity=viscosity_edac,
                                                    density_calculator=ContinuityDensity(),
                                                    pressure_acceleration=nothing,
                                                    acceleration=(0.0, -gravity))


else
    method = "wcsph"
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, clip_negative_pressure=false)
    tank = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

    alpha = 0.02
    viscosity_wcsph = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
    density_diffusion = DensityDiffusionAntuono(tank.fluid, delta=0.1)

    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, ContinuityDensity(),
                                            state_equation, smoothing_kernel,
                                            smoothing_length, viscosity=viscosity_wcsph,
                                            density_diffusion=density_diffusion,
                                            acceleration=(0.0, -gravity), correction=nothing,
                                            surface_tension=nothing,
                                            reference_particle_spacing=0)

end

formatted_string = replace(string(particle_spacing), "." => "")

postprocessing_cb = PostprocessCallback(; dt=0.02, output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, exclude_boundary=false, max_x_coord,
                                        interpolated_pressure_P1, interpolated_pressure_P2, interpolated_pressure_P3,
                                        particle_pressure_P1, particle_pressure_P2, particle_pressure_P3)

# Disable loop flipping to produce consistent results over different thread numbers
boundary_density_calculator = AdamiPressureExtrapolation(allow_loop_flipping=false)


# Save at certain timepoints which allows comparison to the results of Marrone et al.,
# i.e. (1.5, 2.36, 3.0, 5.7, 6.45).
# Please note that the images in Marrone et al. are obtained at a particle_spacing = H/320,
# which takes between 2 and 4 hours.
saving_paper = SolutionSavingCallback(save_times=[0.0, 0.371, 0.584, 0.743, 1.411, 1.597],
                                      prefix="marrone_times")

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=smoothing_length,
              smoothing_kernel=smoothing_kernel,
              boundary_density_calculator=boundary_density_calculator,
              boundary_layers=4,
              state_equation=state_equation,
              solution_prefix="validation_" * method * "_" * formatted_string,
              tspan=tspan,
              fluid_system=fluid_system,
              extra_callback=postprocessing_cb,
              extra_callback2=saving_paper,
              update_strategy=SerialUpdate()) # To get the same results with different thread numbers
