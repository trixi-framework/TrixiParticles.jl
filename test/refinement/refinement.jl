@testset verbose=true "Refinement" begin
    include("variable_smoothing_length.jl")
    include("interactions.jl")
    include("edac.jl")
    include("resize.jl")
end
