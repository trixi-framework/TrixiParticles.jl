# Separate file that can be executed to only run unit tests.
# Include `test_util.jl` first.
@testset verbose=true "Unit Tests" begin
    include("callbacks/callbacks.jl")
    include("general/general.jl")
    include("setups/setups.jl")
    include("systems/systems.jl")
    include("schemes/schemes.jl")
end;
