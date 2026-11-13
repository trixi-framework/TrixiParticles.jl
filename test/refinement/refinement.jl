@testset verbose=true "Refinement" begin
    include("variable_smoothing_length.jl")
    include("interactions.jl")
    include("edac.jl")
    include("resize.jl")
    include("refinement_criteria.jl")
    include("spacing.jl")
    include("splitting.jl")
end
