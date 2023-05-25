# Separate file that can be executed to only run unit tests.
# Include `test_util.jl` first.
@testset verbose=true "Unit Tests" begin
    include("systems/systems.jl")
    include("general/general.jl")
    include("schemes/schemes.jl")
    include("setups/setups.jl")
end;
