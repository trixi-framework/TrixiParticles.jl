# Separate file that one can execute to only run unit tests
@testset verbose=true "Unit Tests" begin
    include("containers/containers.jl")
    include("general/general.jl")
    include("schemes/schemes.jl")
    include("setups/setups.jl")
end;
