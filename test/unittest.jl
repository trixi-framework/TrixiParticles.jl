# Separate file that can be executed to only run unit tests.
# Include `test_util.jl` first.
@testset verbose=true "Unit Tests" begin
    using Metal

    Metal.versioninfo()

    x = Metal.zeros(10)
    display(x)
    display(x .+ 1)

    include("callbacks/callbacks.jl")
    include("general/general.jl")
    include("setups/setups.jl")
    include("systems/systems.jl")
    include("schemes/schemes.jl")
    include("preprocessing/preprocessing.jl")
end;
