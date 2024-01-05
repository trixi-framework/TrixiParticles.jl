using TrixiParticles

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"),
              fluid_particle_spacing=0.05,
              initial_fluid_size=(2.0, 1.0, 0.9),
              tank_size=(2.0, 1.0, 1.2),
              acceleration=(0.0, 0.0, -9.81),
              smoothing_kernel=SchoenbergCubicSplineKernel{3}())
