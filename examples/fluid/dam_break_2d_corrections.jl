using TrixiParticles

fluid_density = 1000.0

particle_spacing = 0.05

tspan = (0.0, 5.7 / sqrt(9.81))

correction_dict = Dict(
    "no_correction" => Nothing(),
    "shepard_kernel_correction" => ShepardKernelCorrection(),
    "akinci_free_surf_correction" => AkinciFreeSurfaceCorrection(fluid_density),
    "kernel_gradient_summation_correction" => KernelCorrection(),
    "kernel_gradient_continuity_correction" => KernelCorrection(),
    "blended_gradient_summation_correction" => BlendedGradientCorrection(0.5),
    "blended_gradient_continuity_correction" => BlendedGradientCorrection(0.2),
    "gradient_summation_correction" => GradientCorrection(),
    "mixed_kernel_gradient_summation_correction" => MixedKernelGradientCorrection(),
    # "gradient_continuity_correction" => GradientCorrection(),
    # "mixed_kernel_gradient_continuity_correction" => MixedKernelGradientCorrection(),
)

smoothing_length_dict = Dict(
    "no_correction" => 3.0 * particle_spacing,
    "shepard_kernel_correction" => 3.0 * particle_spacing,
    "akinci_free_surf_correction" => 3.0 * particle_spacing,
    "kernel_gradient_summation_correction" => 4.0 * particle_spacing,
    "kernel_gradient_continuity_correction" => 3.5 * particle_spacing,
    "blended_gradient_summation_correction" => 3.0 * particle_spacing,
    "blended_gradient_continuity_correction" => 4.0 * particle_spacing,
    "gradient_summation_correction" => 3.5 * particle_spacing,
    "mixed_kernel_gradient_summation_correction" => 3.5 * particle_spacing,
    "gradient_continuity_correction" => 4.5 * particle_spacing,
    "mixed_kernel_gradient_continuity_correction" => 4.0 * particle_spacing,
)

density_calculator_dict = Dict(
    "no_correction" => SummationDensity(),
    "shepard_kernel_correction" => SummationDensity(),
    "akinci_free_surf_correction" => SummationDensity(),
    "kernel_gradient_summation_correction" => SummationDensity(),
    "kernel_gradient_continuity_correction" => ContinuityDensity(),
    "blended_gradient_summation_correction" => SummationDensity(),
    "blended_gradient_continuity_correction" => ContinuityDensity(),
    "gradient_summation_correction" => SummationDensity(),
    "gradient_continuity_correction" => ContinuityDensity(),
    "mixed_kernel_gradient_summation_correction" => SummationDensity(),
    "mixed_kernel_gradient_continuity_correction" => ContinuityDensity(),
)

smoothing_kernel_dict = Dict(
    "no_correction" => WendlandC2Kernel{2}(),
    "shepard_kernel_correction" => WendlandC2Kernel{2}(),
    "akinci_free_surf_correction" => WendlandC2Kernel{2}(),
    "kernel_gradient_summation_correction" => WendlandC6Kernel{2}(),
    "kernel_gradient_continuity_correction" => WendlandC6Kernel{2}(),
    "blended_gradient_summation_correction" => WendlandC2Kernel{2}(),
    "blended_gradient_continuity_correction" => WendlandC6Kernel{2}(),
    "gradient_summation_correction" => WendlandC6Kernel{2}(),
    "gradient_continuity_correction" => WendlandC6Kernel{2}(),
    "mixed_kernel_gradient_summation_correction" => WendlandC6Kernel{2}(),
    "mixed_kernel_gradient_continuity_correction" => WendlandC6Kernel{2}(),
)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=3.0 * particle_spacing,
              boundary_density_calculator=ContinuityDensity(),
              fluid_density_calculator=ContinuityDensity(),
              correction=nothing, use_reinit=true,
              prefix="continuity_reinit", tspan=tspan,
              fluid_density=fluid_density, density_diffusion=nothing)

for correction_name in keys(correction_dict)
    local fluid_density_calculator = density_calculator_dict[correction_name]
    local correction = correction_dict[correction_name]
    local smoothing_kernel = smoothing_kernel_dict[correction_name]
    local smoothing_length = smoothing_length_dict[correction_name]

    println("="^100)
    println("fluid/dam_break_2d.jl with ", correction_name)

    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                  fluid_particle_spacing=particle_spacing,
                  smoothing_length=smoothing_length,
                  boundary_density_calculator=SummationDensity(),
                  fluid_density_calculator=fluid_density_calculator,
                  correction=correction, use_reinit=false,
                  clip_negative_pressure=(fluid_density_calculator isa SummationDensity),
                  smoothing_kernel=smoothing_kernel,
                  prefix="$(correction_name)", tspan=tspan,
                  fluid_density=fluid_density, density_diffusion=Nothing(),
                  boundary_layers=5)
end
