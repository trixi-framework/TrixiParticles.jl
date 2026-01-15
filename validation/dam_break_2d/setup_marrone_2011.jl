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

using TrixiParticles
using TrixiParticles.JSON

# Include custom quantity functions for sensor interpolation
include("sensors.jl")

# `particle_spacing` in this case is set relative to `H`, the initial height of the fluid.
# Use `particles_per_height = 80` or `320` for validation.
# Marrone et al. (2011) used `particles_per_height = 320` for the images of the wave front.
# De Courcy et al. (2024) used `particles_per_height = 400` for the pressure sensor plots.
# Note: `particles_per_height >= 320` takes a few hours!
particles_per_height = 40
H = 0.6
particle_spacing = H / particles_per_height

# Import variables from the example file
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              alpha=0.02, # This is used by Marrone et al. (2011)
              sol=nothing, ode=nothing, fluid_particle_spacing=particle_spacing)

use_edac = false # Set to false to use WCSPH

tspan = (0.0, 6.45 / sqrt(gravity / H))

fluid_density = 1000.0

# Marrone et al. (2011) used a sound speed of 40 * sqrt(g * H) for WCSPH,
# but found that 20 * sqrt(g * H) yielded similar results at lower cost.
# De Courcy et al. (2024) used a sound speed of 50 * sqrt(g * H) for WCSPH.
sound_speed = 20 * sqrt(gravity * H)

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

if use_edac
    method = "edac"
    state_equation = nothing

    tank_edac = RectangularTank(particle_spacing, initial_fluid_size, tank_size,
                                fluid_density,
                                n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                                acceleration=(0.0, -gravity), state_equation=nothing)

    fluid_system = EntropicallyDampedSPHSystem(tank_edac.fluid, smoothing_kernel,
                                               smoothing_length,
                                               sound_speed, viscosity=viscosity_fluid,
                                               density_calculator=ContinuityDensity(),
                                               pressure_acceleration=nothing,
                                               acceleration=(0.0, -gravity))

else
    method = "wcsph"
end

formatted_string = string(particles_per_height)

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
                                                 sqrt(gravity / H),
                                      prefix="marrone_times")

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              boundary_density_calculator=boundary_density_calculator,
              state_equation=state_equation,
              solution_prefix="validation_" * method * "_" * formatted_string,
              tspan=tspan, fluid_system=fluid_system,
              update_strategy=nothing,
              extra_callback=postprocessing_cb,
              extra_callback2=saving_paper)
