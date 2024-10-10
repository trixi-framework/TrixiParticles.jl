# 3D channel flow simulation with open boundaries.
using TrixiParticles
using OrdinaryDiffEq

# load variables
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "pipe_flow_2d.jl"),
              sol=nothing)

function velocity_function3d(pos, t)
    # Use this for a time-dependent inflow velocity
    # return SVector(0.5prescribed_velocity * sin(2pi * t) + prescribed_velocity, 0)

    return SVector(prescribed_velocity, 0.0, 0.0)
end

domain_size = (1.0, 0.4, 0.4)

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2], domain_size[3])

pipe3d = RectangularTank(particle_spacing, domain_size, boundary_size, fluid_density,
                         pressure=pressure, n_layers=boundary_layers,
                         faces=(false, false, true, true, true, true))

flow_direction = [1.0, 0.0, 0.0]

# setup simulation
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "pipe_flow_2d.jl"),
              domain_size=domain_size, flow_direction=flow_direction,
              pipe=pipe3d,
              n_buffer_particles=4 * pipe3d.n_particles_per_dimension[2]^2,
              smoothing_kernel=WendlandC2Kernel{3}(),
              reference_velocity=velocity_function3d,
              inflow=InFlow(;
                            plane=([0.0, 0.0, 0.0],
                                   [0.0, domain_size[2], 0.0],
                                   [0.0, 0.0, domain_size[3]]), flow_direction,
                            open_boundary_layers, density=fluid_density, particle_spacing),
              outflow=OutFlow(;
                              plane=([domain_size[1], 0.0, 0.0],
                                     [domain_size[1], domain_size[2], 0.0],
                                     [domain_size[1], 0.0, domain_size[3]]),
                              flow_direction, open_boundary_layers, density=fluid_density,
                              particle_spacing))
