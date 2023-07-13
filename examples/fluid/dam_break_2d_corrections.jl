using TrixiParticles

particle_spacing = 0.05
smoothing_length = 1.15 * particle_spacing

water_density = 1000.0

boundary_density_calculator = SummationDensity()

correction_dict = Dict("no_correction" => Nothing(),
                       "shepard_kernel_correction" => ShepardKernelCorrection(),
                       "akinci_free_surf_correction" => AkinciFreeSurfaceCorrection(water_density),
                       "kernel_gradient_summation_correction" => KernelGradientCorrection(),
                       "kernel_gradient_continuity_correction" => KernelGradientCorrection())

density_calculator_dict = Dict("no_correction" => SummationDensity(),
                               "shepard_kernel_correction" => SummationDensity(),
                               "akinci_free_surf_correction" => SummationDensity(),
                               "kernel_gradient_summation_correction" => SummationDensity(),
                               "kernel_gradient_continuity_correction" => ContinuityDensity())

for correction_name in keys(correction_dict)
    fluid_density_calculator = density_calculator_dict[correction_name]
    correction = correction_dict[correction_name]
    # prefix?

    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                  particle_spacing=particle_spacing, smoothing_length=smoothing_length,
                  boundary_density_calculator=boundary_density_calculator,
                  fluid_density_calculator=fluid_density_calculator,
                  correction=correction)
end
