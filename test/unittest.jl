# Separate file that can be executed to only run unit tests.
# Include `test_util.jl` first.
@testset verbose=true "Unit Tests" begin
    include("callbacks/callbacks.jl")
    include("general/general.jl")
    include("custom_quantities/custom_quantities.jl")
    include("custom_quantities/mechanical_work_calculator.jl")
    include("custom_quantities/structure_motion_calculator.jl")
    include("setups/setups.jl")
    include("systems/systems.jl")
    include("schemes/schemes.jl")
    include("preprocessing/preprocessing.jl")
    include("io/write_vtk.jl")
    include("io/read_vtk.jl")
end;
