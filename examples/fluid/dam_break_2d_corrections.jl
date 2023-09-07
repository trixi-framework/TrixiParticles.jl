using TrixiParticles

fluid_density = 1000.0

particle_spacing = 0.05
smoothing_length = 1.15 * particle_spacing

boundary_density_calculator = SummationDensity()

correction_dict = Dict(
    "no_correction" => Nothing(),
    "shepard_kernel_correction" => ShepardKernelCorrection(),
    "akinci_free_surf_correction" => AkinciFreeSurfaceCorrection(fluid_density),
    "kernel_gradient_summation_correction" => KernelGradientCorrection(),
    "kernel_gradient_continuity_correction" => KernelGradientCorrection(),
)

density_calculator_dict = Dict(
    "no_correction" => SummationDensity(),
    "shepard_kernel_correction" => SummationDensity(),
    "akinci_free_surf_correction" => SummationDensity(),
    "kernel_gradient_summation_correction" => SummationDensity(),
    "kernel_gradient_continuity_correction" => ContinuityDensity(),
)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              particle_spacing=particle_spacing, smoothing_length=smoothing_length,
              boundary_density_calculator=ContinuityDensity(),
              fluid_density_calculator=ContinuityDensity(),
              correction=Nothing(), use_reinit=true,
              relaxation_step_file_prefix="relaxation_continuity_reinit",
              simulation_step_file_prefix="continuity_reinit")

for correction_name in keys(correction_dict)
    fluid_density_calculator = density_calculator_dict[correction_name]
    correction = correction_dict[correction_name]

    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                  particle_spacing=particle_spacing, smoothing_length=smoothing_length,
                  boundary_density_calculator=boundary_density_calculator,
                  fluid_density_calculator=fluid_density_calculator,
                  correction=correction, use_reinit=false,
                  relaxation_step_file_prefix="relaxation_$(correction_name)",
                  simulation_step_file_prefix="$(correction_name)")
end
