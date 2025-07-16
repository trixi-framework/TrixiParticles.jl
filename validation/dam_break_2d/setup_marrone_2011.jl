# 2D dam break validation setup based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using TrixiParticles
using JSON

include("sensors.jl")

# Import variables from the example file
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, ode=nothing)

use_edac = false # Set to false to use WCSPH

tspan = (0.0, 7 / sqrt(gravity / H))

# `particle_spacing` in this case is set relative to `H`, the initial height of the fluid.
# Use H / 80, H / 320 for validation.
# Note: H / 320 takes a few hours!
particles_per_height = 200
particle_spacing = H / particles_per_height
# For reference:
# H/320 is 0.001875m
# H/80 is 0.0075m
# H/40 is 0.015m

fluid_density = 1000.0
sound_speed = 50 * sqrt(gravity * H)

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

pressure_P1 = (system, v_ode, u_ode, semi,
                            t) -> interpolated_pressure([tank_right_wall_x, P1_y_top],
                                                        [tank_right_wall_x, P1_y_bottom],
                                                        v_ode, u_ode, t, system, semi)
pressure_P2 = (system, v_ode, u_ode, semi,
                            t) -> interpolated_pressure([tank_right_wall_x, P2_y_top],
                                                        [tank_right_wall_x, P2_y_bottom],
                                                        v_ode, u_ode, t, system, semi)
pressure_P3 = (system, v_ode, u_ode, semi,
                            t) -> interpolated_pressure([tank_right_wall_x, P3_y_top],
                                                        [tank_right_wall_x, P3_y_bottom],
                                                        v_ode, u_ode, t, system, semi)
pressure_P4 = (system, v_ode, u_ode, semi,
                            t) -> interpolated_pressure([tank_right_wall_x, P4_y_top],
                                                        [tank_right_wall_x, P4_y_bottom],
                                                        v_ode, u_ode, t, system, semi)

if use_edac
    method = "edac"
    state_equation = nothing

    tank_edac = RectangularTank(particle_spacing, initial_fluid_size, tank_size,
                                fluid_density,
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
end

formatted_string = replace(string(particle_spacing), "." => "")

postprocessing_cb = PostprocessCallback(; dt=0.01 / sqrt(gravity / H), output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, exclude_boundary=false,
                                        max_x_coord,
                                        pressure_P1, pressure_P2,
                                        pressure_P3, pressure_P4)

# Disable loop flipping to produce consistent results over different thread numbers
boundary_density_calculator = AdamiPressureExtrapolation(allow_loop_flipping=false)

# Save at certain timepoints which allows comparison to the results of Marrone et al.,
# i.e. (1.5, 2.36, 3.0, 5.7, 6.45).
# Please note that the images in Marrone et al. are obtained at a particle_spacing = H/320,
# which takes between 2 and 4 hours.
saving_paper = SolutionSavingCallback(save_times=[0.0, 1.5, 3.0, 5.7, 6.45] ./ sqrt(gravity / H),
                                      prefix="marrone_times")

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              boundary_density_calculator=boundary_density_calculator,
              state_equation=state_equation,
              solution_prefix="validation_" * method * "_" * formatted_string,
              tspan=tspan,
            #   fluid_system=fluid_system,
              extra_callback=postprocessing_cb,
              extra_callback2=saving_paper)
            #   update_strategy=SerialUpdate()) # To get the same results with different thread numbers
