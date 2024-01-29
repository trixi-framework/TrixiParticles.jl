@doc raw"""
    TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle)

Trivial neighborhood search that simply loops over all particles.
"""
struct TrivialNeighborhoodSearch{NDIMS, ELTYPE, EP, PB}
    search_radius :: ELTYPE
    eachparticle  :: EP
    periodic_box  :: PB

    function TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle;
                                              min_corner=nothing,
                                              max_corner=nothing) where {NDIMS}
        if (min_corner === nothing && max_corner === nothing) || search_radius < eps()
            # No periodicity
            periodic_box = nothing
        elseif min_corner !== nothing && max_corner !== nothing
            periodic_box = PeriodicBox(min_corner, max_corner)
        else
            throw(ArgumentError("`min_corner` and `max_corner` must either be " *
                                "both `nothing` or both an array or tuple"))
        end

        new{NDIMS, typeof(search_radius),
            typeof(eachparticle), typeof(periodic_box)}(search_radius, eachparticle,
                                                        periodic_box)
    end
end

@inline function Base.ndims(neighborhood_search::TrivialNeighborhoodSearch{NDIMS}) where {
                                                                                          NDIMS
                                                                                          }
    return NDIMS
end

@inline initialize!(search::TrivialNeighborhoodSearch, coords_fun) = search
@inline update!(search::TrivialNeighborhoodSearch, coords_fun) = search
@inline eachneighbor(coords, search::TrivialNeighborhoodSearch) = search.eachparticle
