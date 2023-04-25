include("boundary/boundary.jl")
include("fluid/WCSPH.jl")
include("solid/solid.jl")

function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container, neighbor_container)
    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps() < distance2 <=
               larger_compact_support(particle_container, neighbor_container)^2

                distance = sqrt(distance2)

                interaction!(particle_container, neighbor_container,
                             dv, v_particle_container, v_neighbor_container,
                             particle, neighbor, pos_diff, distance)
            end
        end
    end

    return dv
end
