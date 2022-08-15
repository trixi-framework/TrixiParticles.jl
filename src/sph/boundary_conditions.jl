# Create Tuple of BCs from single BC
digest_boundary_conditions(boundary_condition) = (boundary_condition, )
digest_boundary_conditions(boundary_condition::Tuple) = boundary_condition
digest_boundary_conditions(::Nothing) = ()


struct BoundaryConditionMonaghanKajtar{ELTYPE<:Real, NS}
    K                   ::ELTYPE
    coordinates         ::Array{ELTYPE, 2}
    mass                ::Vector{ELTYPE}
    spacing             ::Vector{ELTYPE} # 1/β in Monaghan, Kajtar (2009). TODO ELTYPE or hardcoded float?
    neighborhood_search ::NS

    function BoundaryConditionMonaghanKajtar(K, coordinates, masses, spacings; neighborhood_search=nothing)
        new{typeof(K), typeof(neighborhood_search)}(K, coordinates, masses, spacings, neighborhood_search)
    end
end

function initialize!(boundary_conditions::BoundaryConditionMonaghanKajtar, semi)
    @unpack neighborhood_search = boundary_conditions

    initialize!(neighborhood_search, boundary_conditions,
                semi, particles=eachparticle(boundary_conditions))
end

@inline nparticles(boundary_container::BoundaryConditionMonaghanKajtar) = length(boundary_container.mass)
