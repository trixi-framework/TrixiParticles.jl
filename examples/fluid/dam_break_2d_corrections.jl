using TrixiParticles

fluid_density = 1000.0
atmospheric_pressure = 100000.0
gravity = 9.81

initial_fluid_size = (2.0, 1.0)
sound_speed = 20 * sqrt(gravity * initial_fluid_size[2])

particle_spacing = 0.05
smoothing_length = 1.15 * particle_spacing
tspan = (0.0, 5.7 / sqrt(gravity))

boundary_density_calculator = SummationDensity()

correction_dict = Dict(
    "no_correction" => Nothing(),
    "shepard_kernel_correction" => ShepardKernelCorrection(),
    "akinci_free_surf_correction" => AkinciFreeSurfaceCorrection(fluid_density),
    "kernel_gradient_summation_correction" => KernelGradientCorrection(),
    "kernel_gradient_continuity_correction" => KernelGradientCorrection(),
    "blended_gradient_summation_correction" => BlendedGradientCorrection(0.5),
    "blended_gradient_continuity_correction" => BlendedGradientCorrection(0.1),
)

density_calculator_dict = Dict(
    "no_correction" => SummationDensity(),
    "shepard_kernel_correction" => SummationDensity(),
    "akinci_free_surf_correction" => SummationDensity(),
    "kernel_gradient_summation_correction" => SummationDensity(),
    "kernel_gradient_continuity_correction" => ContinuityDensity(),
    "blended_gradient_summation_correction" => SummationDensity(),
    "blended_gradient_continuity_correction" => ContinuityDensity(),
)

smoothing_kernel_dict = Dict(
    "no_correction" => WendlandC2Kernel{2}(),
    "shepard_kernel_correction" => WendlandC2Kernel{2}(),
    "akinci_free_surf_correction" => WendlandC2Kernel{2}(),
    "kernel_gradient_summation_correction" => WendlandC2Kernel{2}(),
    "kernel_gradient_continuity_correction" => WendlandC2Kernel{2}(),
    "blended_gradient_summation_correction" => WendlandC2Kernel{2}(),
    "blended_gradient_continuity_correction" => WendlandC2Kernel{2}(),
)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing, smoothing_length=smoothing_length,
              boundary_density_calculator=ContinuityDensity(),
              fluid_density_calculator=ContinuityDensity(),
              correction=Nothing(), use_reinit=true,
              prefix="continuity_reinit", tspan=tspan)

# Clip negative pressure to be able to use `SummationDensity`
state_equation = StateEquationCole(sound_speed, 7, fluid_density, atmospheric_pressure,
                                   background_pressure=atmospheric_pressure,
                                   clip_negative_pressure=true)

for correction_name in keys(correction_dict)
    local fluid_density_calculator = density_calculator_dict[correction_name]
    local correction = correction_dict[correction_name]
    local smoothing_kernel = smoothing_kernel_dict[correction_name]

    println("="^100)
    println("fluid/dam_break_2d.jl with ", correction_name)

    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                  fluid_particle_spacing=particle_spacing,
                  smoothing_length=smoothing_length,
                  boundary_density_calculator=boundary_density_calculator,
                  fluid_density_calculator=fluid_density_calculator,
                  correction=correction, use_reinit=false,
                  state_equation=state_equation, smoothing_kernel=smoothing_kernel,
                  prefix="$(correction_name)", tspan=tspan)
end
