# ==========================================================================================
# 3D Pipe Flow Simulation with Open Boundaries (Inflow/Outflow)
#
# This example extends the 2D pipe flow simulation (`pipe_flow_2d.jl`) to three
# dimensions.
# ==========================================================================================

using TrixiParticles

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.05

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 3

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 6

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

function velocity_function3d(pos, t)
    # Use this for a time-dependent inflow velocity
    # return SVector(0.5prescribed_velocity * sin(2pi * t) + prescribed_velocity, 0)

    return SVector(prescribed_velocity, 0.0, 0.0)
end

domain_size = (1.0, 0.4, 0.4)

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2], domain_size[3])

flow_direction = [1.0, 0.0, 0.0]

# setup simulation
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "pipe_flow_2d.jl"),
              domain_size=domain_size, boundary_size=boundary_size,
              flow_direction=flow_direction, faces=(false, false, true, true, true, true),
              tspan=tspan, reference_velocity=velocity_function3d,
              open_boundary_layers=open_boundary_layers,
              plane_in=([0.0, 0.0, 0.0], [0.0, domain_size[2], 0.0],
                        [0.0, 0.0, domain_size[3]]),
              plane_out=([domain_size[1], 0.0, 0.0], [domain_size[1], domain_size[2], 0.0],
                         [domain_size[1], 0.0, domain_size[3]]))
