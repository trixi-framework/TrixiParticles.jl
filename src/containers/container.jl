
initialize!(container, neighborhood_search) = container
update!(container, container_index, v, u, v_ode, u_ode, semi, t) = container

include("fluid_container.jl")
include("solid_container.jl")
include("boundary_container.jl") # This depends on fluid and solid containers
