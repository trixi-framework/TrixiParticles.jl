using TrixiParticles

# The time estimated is relative to an average performance CPU core
# in 2025 based on CPUMark data 2.1k (single thread).

# TRIXIP: SEQUENTIAL
# TIME ESTIMATE: 20m
trixi_include(@__MODULE__,
              joinpath(validation_dir(),
                       "hydrostatic_water_column_2d",
                       "validation.jl"), tspan=(0.0, 0.3),
              n_particles_plate_y=3)

# TRIXIP: SEQUENTIAL
# TIME ESTIMATE: 800m
trixi_include(@__MODULE__,
              joinpath(validation_dir(),
                       "hydrostatic_water_column_2d",
                       "validation.jl"), tspan=(0.0, 0.5),
              n_particles_plate_y=5)

# TIME ESTIMATE: 7200m
trixi_include(@__MODULE__,
              joinpath(validation_dir(),
                       "hydrostatic_water_column_2d",
                       "validation.jl"), tspan=(0.0, 0.5),
              n_particles_plate_y=11)

# TIME ESTIMATE: 10400m
trixi_include(@__MODULE__,
              joinpath(validation_dir(),
                       "hydrostatic_water_column_2d",
                       "validation.jl"), tspan=(0.0, 0.5),
              n_particles_plate_y=13)
