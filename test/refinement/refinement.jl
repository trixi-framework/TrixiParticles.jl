@testset verbose=true "Refinement" begin
    include("variable_smoothing_length.jl")
    include("interactions.jl")
    include("edac.jl")
    include("resize.jl")
    include("search_radius.jl")
    include("callback.jl")
    include("open_boundary.jl")
    include("refinement_criteria.jl")
    include("spacing.jl")
    include("splitting.jl")
    include("merging.jl")
    include("shifting.jl")
end
