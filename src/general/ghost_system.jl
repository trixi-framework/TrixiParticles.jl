struct GhostSystem{S, NDIMS, IC, U, V} <: System{NDIMS, IC}
    system::S
    v::V
    u::U
    buffer::Nothing

    function GhostSystem(system)
        v = zeros(eltype(system), v_nvariables(system), nparticles(system))
        u = zeros(eltype(system), u_nvariables(system), nparticles(system))
        return new{typeof(system), ndims(system),
                   typeof(system.initial_condition),
                   typeof(v), typeof(u)}(system, v, u, nothing)
    end
end

vtkname(system::GhostSystem) = "ghost"
timer_name(::GhostSystem) = "ghost"

@inline nparticles(system::GhostSystem) = nparticles(system.system)
@inline Base.eltype(system::GhostSystem) = eltype(system.system)

@inline u_nvariables(system::GhostSystem) = 0
@inline v_nvariables(system::GhostSystem) = 0

@inline compact_support(system, neighbor::GhostSystem) = compact_support(system, neighbor.system)
@inline compact_support(system::GhostSystem, neighbor) = compact_support(system.system, neighbor)
@inline compact_support(system::GhostSystem, neighbor::GhostSystem) = compact_support(system.system, neighbor.system)

@inline initial_coordinates(system::GhostSystem) = initial_coordinates(system.system)
@inline current_coordinates(u, system::GhostSystem) = current_coordinates(system.u, system.system)

function write_u0!(u0, system::GhostSystem)
    return u0
end

function write_v0!(v0, system::GhostSystem)
    return v0
end

function partition(system::GhostSystem, neighborhood_search)
    # TODO variable size
    keep = trues(nparticles(system))
    for particle in eachparticle(system)
        cell = PointNeighbors.cell_coords(initial_coords(system, particle), neighborhood_search)
        mpi_neighbors = neighborhood_search.cell_list.neighbor_cells[2]
        # length of mpi_neighbors is the number of mirror cells. Everything after that is ghost.
        if neighborhood_search.cell_list.cell_indices[cell...] <= length(mpi_neighbors)
            # Remove particle from this rank
            keep[particle] = false
        end
    end

    return GhostSystem(keep_particles(system.system, keep))
end

function copy_neighborhood_search_ghost(nhs, search_radius, n_points)
    (; periodic_box) = nhs

    cell_list = copy_cell_list_ghost(nhs.cell_list, search_radius, periodic_box)

    return GridNeighborhoodSearch{ndims(nhs)}(; search_radius, n_points, periodic_box,
                                              cell_list,
                                              update_strategy = nhs.update_strategy)
end

function copy_cell_list_ghost(cell_list, search_radius, periodic_box)
    (; min_corner, max_corner) = cell_list

    return PointNeighbors.P4estCellList(; min_corner, max_corner, search_radius,
                                        backend = typeof(cell_list.cells), ghost=true)
end

function create_neighborhood_search(neighborhood_search, system, neighbor::GhostSystem)
    return copy_neighborhood_search_ghost(neighborhood_search, compact_support(system, neighbor),
                                    nparticles(neighbor))
end

function update_nhs!(neighborhood_search,
                     system, neighbor::GhostSystem,
                     u_system, u_neighbor)
end

function update_nhs!(neighborhood_search,
                     system::GhostSystem, neighbor,
                     u_system, u_neighbor)
end

function update_nhs!(neighborhood_search,
                     system::GhostSystem, neighbor::GhostSystem,
                     u_system, u_neighbor)
end

# function mpi_communication1!(system::GhostSystem, v, u, v_ode, u_ode, semi, t)
#     # Receive data from other ranks
#     return system
# end

function mpi_communication3!(system::GhostSystem, v, u, v_ode, u_ode, semi, t)
    # TODO Receive only density and pressure from other ranks
    # and move the rest to `mpi_communication1!`.
    n_particles_from_rank = zeros(Int, MPI.Comm_size(MPI.COMM_WORLD))
    for source in 0:MPI.Comm_size(MPI.COMM_WORLD) - 1
        buffer = view(n_particles_from_rank, source + 1)
        MPI.Recv!(buffer, MPI.COMM_WORLD; source=source, tag=0)
    end

    @info "" n_particles_from_rank
    total_particles = sum(n_particles_from_rank)

    requests = Vector{MPI.Request}(undef, 3 * MPI.Comm_size(MPI.COMM_WORLD))

    coordinates = zeros(ndims(system), total_particles)
    for source in 0:MPI.Comm_size(MPI.COMM_WORLD) - 1
        buffer = view(coordinates, :, (1 + sum(n_particles_from_rank[1:source])):total_particles)
        # requests[source + 1] = MPI.Irecv!(buffer, MPI.COMM_WORLD; source=source, tag=1)
        MPI.Recv!(buffer, MPI.COMM_WORLD; source=source, tag=1)
    end

    # Receive density and pressure from other ranks
    density = zeros(total_particles)
    for source in 0:MPI.Comm_size(MPI.COMM_WORLD) - 1
        buffer = view(density, (1 + sum(n_particles_from_rank[1:source])):total_particles)
        # requests[source + MPI.Comm_size(MPI.COMM_WORLD) + 1] = MPI.Irecv!(buffer, MPI.COMM_WORLD; source=source, tag=2)
        MPI.Recv!(buffer, MPI.COMM_WORLD; source=source, tag=2)
    end

    pressure = zeros(total_particles)
    for source in 0:MPI.Comm_size(MPI.COMM_WORLD) - 1
        buffer = view(pressure, (1 + sum(n_particles_from_rank[1:source])):total_particles)
        # requests[source + 2 * MPI.Comm_size(MPI.COMM_WORLD) + 1] = MPI.Irecv!(buffer, MPI.COMM_WORLD; source=source, tag=3)
        MPI.Recv!(buffer, MPI.COMM_WORLD; source=source, tag=3)
    end

    # MPI.Waitall(requests)

    return system
end

function update_quantities!(system::GhostSystem, v, u,
                            v_ode, u_ode, semi, t)
    return system
end

function update_pressure!(system::GhostSystem, v, u, v_ode, u_ode, semi, t)
    return system
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::GhostSystem,
                   neighbor_system)
    return dv
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::GhostSystem,
                   neighbor_system::GhostSystem)
    return dv
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system,
                   neighbor_system::GhostSystem)
    interact!(dv, v_particle_system, u_particle_system,
              v_neighbor_system, u_neighbor_system, neighborhood_search,
              particle_system, neighbor_system.system)

    return dv
end

@inline add_velocity!(du, v, particle, system::GhostSystem) = du
