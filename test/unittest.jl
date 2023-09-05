# Separate file that can be executed to only run unit tests.
# Include `test_util.jl` first.
@testset verbose=true "Unit Tests" begin
    include("general/general.jl")
    include("neighborhood_search/neighborhood_search.jl")
    include("setups/setups.jl")
    include("systems/systems.jl")
    include("schemes/schemes.jl")
end;
