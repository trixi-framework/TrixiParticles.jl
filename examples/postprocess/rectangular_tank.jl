using TrixiParticles

# Any function can be implemented and will be called after each timestep! See example below:
a = function(pp, t, system, u, v, system_name) println("test_func ", t) end
example_cb = PostprocessCallback([a,])

pp_cb = PostprocessCallback([   a,
                                TrixiParticles.calculate_ekin,
                                TrixiParticles.calculate_total_mass,
                                TrixiParticles.max_pressure,
                                TrixiParticles.min_pressure,
                                TrixiParticles.avg_pressure,
                                TrixiParticles.max_density,
                                TrixiParticles.min_density,
                                TrixiParticles.avg_density,
                            ])

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"),
              pp_callback=pp_cb)
