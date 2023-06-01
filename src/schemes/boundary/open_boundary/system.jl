
struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, C, NDIMS2} <: System{NDIMS}
    initial_condition        :: InitialCondition{ELTYPE}
    mass                     :: Array{ELTYPE, 1} # [particle]
    pressure                 :: Array{ELTYPE, 1} # [particle]
    J                        :: Array{ELTYPE, 2} # [J, particle]
    previous_characteristics :: Array{ELTYPE, 2} # [J, particle]
    boundary_zone            :: BZ
    in_domain                :: BitVector
    interior_system          :: System
    zone_origin              :: SVector{NDIMS, ELTYPE}
    zone                     :: SMatrix{NDIMS, NDIMS, ELTYPE, NDIMS2}
    normal_vector            :: SVector{NDIMS, ELTYPE}
    acceleration             :: SVector{NDIMS, ELTYPE}
    cache                    :: C

    function OpenBoundarySPHSystem(initial_condition, boundary_zone, zone_limits,
                                   zone_origin, interior_system)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)
        n_particles = nparticles(initial_condition)

        mass = copy(initial_condition.mass)
        pressure = Vector{ELTYPE}(undef, n_particles)

        J = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = Array{ELTYPE, 2}(undef, 3, length(mass))
        in_domain = trues(length(mass))

        cache = create_cache(n_particles, ELTYPE, interior_system.density_calculator)

        zone_origin_ = SVector{NDIMS}(zone_origin)
        zone = zeros(NDIMS, NDIMS)

        # TODO either check if the vectores are perpendicular to the faces, or obtain perpendicular
        # vectors by using the cross-product (?):
        # zone[1, :] = cross(zone[2, :], zone[3, :])
        # zone[2, :] = cross(zone[1, :], zone[3, :])
        # zone[3, :] = cross(zone[1, :], zone[2, :])
        for dim in 1:NDIMS
            zone[dim, :] = zone_limits[dim] - zone_origin_
        end

        zone_ = SMatrix{NDIMS, NDIMS}(zone)
        normal_vector_ = SVector{NDIMS}(normalize(zone_[1, :]))

        return new{typeof(boundary_zone), NDIMS, ELTYPE,
                   typeof(cache), NDIMS^2}(initial_condition, mass, pressure, J,
                                           previous_characteristics, boundary_zone,
                                           in_domain,
                                           interior_system, zone_origin_, zone_,
                                           normal_vector_,
                                           interior_system.acceleration, cache)
    end
end

struct InFlow end

struct OutFlow end

function (boundary_zone::Union{OutFlow, InFlow})(particle_coords, system)
    @unpack zone, zone_origin = system

    position = particle_coords - zone_origin
    for dim in 1:ndims(system)
        direction = TrixiParticles.current_coords(system.zone, system, dim)
        !(0 <= dot(position, direction) <= dot(direction, direction)) && return false
    end

    return true
end

function update_final!(system, system_index, v, u, v_ode, u_ode, semi, t)
    @unpack boundary_zone = system

    evaluate_characteristics!(system, system_index, u, u_ode, v_ode, semi, boundary_zone)
end

@inline function evaluate_characteristics!(system, system_index, u, u_ode, v_ode, semi,
                                           boundary_zone::InFlow)
end
