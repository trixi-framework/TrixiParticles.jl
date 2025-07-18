using TrixiParticles

# TRIXIP: SEQUENTIAL
trixi_include(@__MODULE__, joinpath(validation_dir(),
                                                "hydrostatic_water_column_2d",
                                                "validation.jl"), tspan=(0.0, 0.3))

# TRIXIP: SEQUENTIAL
trixi_include(@__MODULE__, joinpath(validation_dir(),
                                                "hydrostatic_water_column_2d",
                                                "validation.jl"), tspan=(0.0, 0.5),
                                                n_particles_plate_y=5)

trixi_include(@__MODULE__, joinpath(validation_dir(),
                                                "hydrostatic_water_column_2d",
                                                "validation.jl"), tspan=(0.0, 0.5),
                                                n_particles_plate_y=11)

trixi_include(@__MODULE__, joinpath(validation_dir(),
                                                "hydrostatic_water_column_2d",
                                                "validation.jl"), tspan=(0.0, 0.5),
                                                n_particles_plate_y=13)
