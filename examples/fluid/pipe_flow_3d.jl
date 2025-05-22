# ==========================================================================================
# 3D Pipe Flow Simulation with Open Boundaries (Inflow/Outflow)
#
# This example extends the 2D pipe flow simulation (`pipe_flow_2d.jl`) to three
# dimensions.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters Specific to 3D for overriding 2D Defaults
# ------------------------------------------------------------------------------

particle_spacing_3d = 0.05
boundary_layers_3d = 3
open_boundary_layers_3d = 6 # Open boundary layers (inflow/outflow buffer zones)

# Simulation time span
tspan_3d = (0.0, 2.0)

# Pipe geometry for 3D (length, width, height)
pipe_domain_size_3d = (1.0, 0.4, 0.4)

# Computational domain size for particle generation, including open boundary buffers.
computational_boundary_width_3d = pipe_domain_size_3d[1] +
                                  2 * particle_spacing_3d * open_boundary_layers_3d
computational_boundary_depth_3d = pipe_domain_size_3d[2]
computational_boundary_height_3d = pipe_domain_size_3d[3]
computational_boundary_size_3d = (computational_boundary_width_3d,
                                  computational_boundary_depth_3d,
                                  computational_boundary_height_3d)

# Flow direction in 3D (e.g., along the x-axis)
flow_direction_vector_3d = SVector(1.0, 0.0, 0.0)

# Prescribed inflow velocity magnitude (consistent with 2D example for reuse)
# This `const` needs to be available in the scope where `velocity_function_3d` is defined.
const prescribed_inflow_velocity_magnitude = 2.0

# Velocity profile function for 3D inflow/outflow boundaries
function velocity_function_3d(position, time)
    # Example for time-dependent inflow:
    # return SVector(0.5 * prescribed_inflow_velocity_magnitude * sin(2 * pi * time) +
    #                prescribed_inflow_velocity_magnitude, 0.0, 0.0)
    return SVector(prescribed_inflow_velocity_magnitude, 0.0, 0.0)
end

# SPH smoothing kernel for 3D
smoothing_kernel_3d = WendlandC2Kernel{3}()

# Definition of inflow and outflow planes for 3D.
# For `BoundaryZone`, a plane in 3D can be defined by three non-collinear points.
# A simple way is to define two edges of a rectangle forming the plane.
# Inflow plane (at x=0):
inflow_plane_point1_3d = [0.0, 0.0, 0.0] # Origin corner
inflow_plane_point2_3d = [0.0, pipe_domain_size_3d[2], 0.0] # Edge along y-axis
inflow_plane_point3_3d = [0.0, 0.0, pipe_domain_size_3d[3]] # Edge along z-axis
inflow_plane_definition_3d = (inflow_plane_point1_3d, inflow_plane_point2_3d,
                              inflow_plane_point3_3d)

# Outflow plane (at x=pipe_domain_size_3d[1]):
outflow_plane_point1_3d = [pipe_domain_size_3d[1], 0.0, 0.0]
outflow_plane_point2_3d = [pipe_domain_size_3d[1], pipe_domain_size_3d[2], 0.0]
outflow_plane_point3_3d = [pipe_domain_size_3d[1], 0.0, pipe_domain_size_3d[3]]
outflow_plane_definition_3d = (outflow_plane_point1_3d, outflow_plane_point2_3d,
                               outflow_plane_point3_3d)

# Faces for the `RectangularTank` in 3D: (x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)
# For a pipe, side walls (y and z faces) are solid, ends (x faces) are open.
tank_faces_3d = (false, false, true, true, true, true)

# Buffer size for 3D SPH systems (heuristic)
num_particles_area_3d = round(Int,
                              (pipe_domain_size_3d[2] * pipe_domain_size_3d[3]) /
                              (particle_spacing_3d^2))
sph_system_buffer_size_3d = 4 * num_particles_area_3d

# ------------------------------------------------------------------------------
# Simulation using 2D Base File
# ------------------------------------------------------------------------------

# Include the 2D pipe flow setup. Variables defined above will override defaults.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "pipe_flow_2d.jl"),
              particle_spacing=particle_spacing_3d,
              boundary_layers=boundary_layers_3d,
              open_boundary_layers=open_boundary_layers_3d,
              pipe_domain_size=pipe_domain_size_3d,
              computational_boundary_size=computational_boundary_size_3d,
              faces=tank_faces_3d,
              flow_direction_vector=flow_direction_vector_3d,
              smoothing_kernel=smoothing_kernel_3d,
              sph_system_buffer_size=sph_system_buffer_size_3d,
              reference_velocity=velocity_function_3d,
              inflow_plane=inflow_plane_definition_3d,
              outflow_plane=outflow_plane_definition_3d,
              tspan=tspan_3d)
