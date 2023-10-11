using TrixiParticles

fluid_density = 1000.0

particle_spacing = 0.05
smoothing_length = 1.15 * particle_spacing

gravity = 9.81
relaxation_tspan = (0.0, 3.0)
simulation_tspan = (0.0, 5.7 / sqrt(gravity))

surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.0005,
                                       rho0=fluid_density)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing, smoothing_length=smoothing_length,
              boundary_density_calculator=ContinuityDensity(),
              fluid_density_calculator=SummationDensity(),
              relaxation_step_file_prefix="relaxation_surface_tension",
              simulation_step_file_prefix="surface_tension",
              surface_tension=surface_tension, correction=AkinciFreeSurfaceCorrection(fluid_density),
              relaxation_tspan=relaxation_tspan, simulation_tspan=simulation_tspan)
