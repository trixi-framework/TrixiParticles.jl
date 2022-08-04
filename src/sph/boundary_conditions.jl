# Create Tuple of BCs from single BC
digest_boundary_conditions(boundary_condition) = (boundary_condition, )
digest_boundary_conditions(boundary_condition::Tuple) = boundary_condition
digest_boundary_conditions(::Nothing) = ()


struct BoundaryConditionMonaghanKajtar{ELTYPE<:Real}
    K               ::ELTYPE
    coordinates     ::Array{ELTYPE, 2}
    mass            ::Vector{ELTYPE}
    spacing         ::Vector{ELTYPE} # 1/β in Monaghan, Kajtar (2009). TODO ELTYPE or hardcoded float?

    function BoundaryConditionMonaghanKajtar(K, coordinates, masses, spacings)
        new{typeof(K)}(K, coordinates, masses, spacings)
    end
end

@inline nparticles(boundary_container::BoundaryConditionMonaghanKajtar) = length(boundary_container.mass)
