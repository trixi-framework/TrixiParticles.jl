include("test_util.jl")

const TRIXIPARTICLES_TEST = lowercase(get(ENV, "TRIXIPARTICLES_TEST", "all"))

@testset "All Tests" verbose=true begin
    if TRIXIPARTICLES_TEST in ("all", "unit")
        include("unittest.jl")
    end

    if TRIXIPARTICLES_TEST in ("all", "examples")
        include("examples/examples.jl")
    end
end;
