struct InitialCondition{ELTYPE}
    coordinates :: Array{ELTYPE, 2}
    velocity    :: Array{ELTYPE, 2}
    mass        :: Array{ELTYPE, 1}
    density     :: Array{ELTYPE, 1}
end
