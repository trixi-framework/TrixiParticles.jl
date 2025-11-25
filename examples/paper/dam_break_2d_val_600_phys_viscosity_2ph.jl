# This file computes the pressure sensor data of the dam break setup described in
#
# J. J. De Courcy, T. C. S. Rendall, L. Constantin, B. Titurus, J. E. Cooper.
# "Incompressible δ-SPH via artificial compressibility".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 420 (2024),
# https://doi.org/10.1016/j.cma.2023.116700

using TrixiParticles
using TrixiParticles.JSON
using CUDA

# When using data center CPUs with large numbers of cores, especially on multi-socket
# systems with multiple NUMA nodes, pinning threads to cores can significantly
# improve performance, even for low resolutions.
# using ThreadPinning
# pinthreads(:numa)

#TODO: duplicated
# Size parameters
H = 0.6
# W = 2 * H

# `resolution` in this case is set relative to `H`, the initial height of the fluid.
# Use 40, 80 or 400 for validation.
# Note: 400 takes about 30 minutes on a large data center CPU (much longer with serial update)
resolution = 200

#TODO: duplicated
fluid_particle_spacing = H / resolution

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=fluid_particle_spacing,
              sol=nothing,
              ode=nothing)

# tank_size = (floor(5.366 * H / fluid_particle_spacing) * fluid_particle_spacing, 4.0)


min_corner = (-1.5, -1.5)
max_corner = (6.5, 5.0)
cell_list = FullGridCellList(; min_corner, max_corner, max_points_per_cell=30)

neighborhood_search = GridNeighborhoodSearch{2}(; cell_list, update_strategy=ParallelUpdate())
# neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

# physical values
nu_water = 8.9E-7*10
nu_air = 1.544E-5*10

# switch to physical viscosity model
viscosity_fluid = ViscosityMorris(nu = nu_water)
# viscosity_fluid = ViscosityAdami(nu = 8.9E-7)
#  viscosity_fluid = nothing

# set air viscosity model
viscosity_air = ViscosityMorris(nu = nu_air)

#TODO: duplicated
smoothing_length = 2 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()
fluid_density_calculator = ContinuityDensity()


# ==========================================================================================
# ==== Setup air_system layer

air_size = (tank_size[1], tank_size[2]-H)
air_size2 = (tank_size[1] - W, tank_size[2])
air_density = 1.0

# Air above the initial water
air_system = RectangularShape(fluid_particle_spacing,
                              round.(Int, air_size ./ fluid_particle_spacing),
                              zeros(length(air_size)), density=air_density)

# Air for the rest of the empty volume
air_system2 = RectangularShape(fluid_particle_spacing,
                               round.(Int, air_size2 ./ fluid_particle_spacing),
                               (W, 0.0), density=air_density)

# move on top of the water
for i in axes(air_system.coordinates, 2)
    air_system.coordinates[:, i] .+= [0.0, H]
end

air_system = union(air_system, air_system2)

#TODO: duplicated
sound_speed=50 * sqrt(9.81 * 0.6)
gravity = 9.81

air_eos = StateEquationCole(; sound_speed, reference_density=air_density, exponent=1,
                            clip_negative_pressure=false, background_pressure=10.0)

air_system_system = WeaklyCompressibleSPHSystem(air_system, fluid_density_calculator,
                                                air_eos, smoothing_kernel, smoothing_length,
                                                viscosity=viscosity_air,
                                                acceleration=(0.0, -gravity))

tank_air_density = fill!(similar(tank.boundary.density), air_density)
tank_air_mass = fill!(similar(tank.boundary.mass), air_density*fluid_particle_spacing^2)

air_boundary_model = BoundaryModelDummyParticles(tank_air_density,
                                             tank_air_mass,
                                             state_equation=air_eos,
                                             boundary_density_calculator,
                                             smoothing_kernel,
                                             smoothing_length,
                                             correction=nothing,
                                             reference_particle_spacing=fluid_particle_spacing,
                                             viscosity=viscosity_wall)

air_boundary_system = WallBoundarySystem(tank.boundary, air_boundary_model,
                                     adhesion_coefficient=0.0)


trixi_include(@__MODULE__,
              joinpath(validation_dir(), "dam_break_2d",
                       "setup_marrone_2011.jl"),
              use_edac=false,
              extra_string = "_phys_viscosity_2ph_v2_pa10",
              viscosity_fluid=viscosity_fluid,
              particles_per_height=resolution,
              sound_speed=50 * sqrt(9.81 * 0.6), # This is used by De Courcy et al. (2024) (120)
              tspan=(0.0, 7 / sqrt(9.81 / 0.6)), # This is used by De Courcy et al. (2024)
              #   tspan=(0.0, 0.0001),
              cfl=0.5,
              parallelization_backend=PolyesterBackend(),
              neighborhood_search=neighborhood_search,
              extra_system=air_system_system,
              extra_system2=air_boundary_system)
