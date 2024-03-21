using TrixiParticles

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fsi", "falling_spheres_2d.jl"),
              solid_system_2=nothing, fluid_particle_spacing=0.02,
              initial_fluid_size=(1.0, 0.9), tank_size=(1.0, 1.0),
              tspan=(0.0, 2.0), abstol=1e-6, reltol=1e-3)
