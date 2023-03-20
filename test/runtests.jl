using Test
using TrixiParticles
using LinearAlgebra
using Printf

include("test_util.jl")

const TRIXIPARTICLES_TEST = lowercase(get(ENV, "TRIXIPARTICLES_TEST", "all"))

@testset "All Tests" verbose=true begin
    if TRIXIPARTICLES_TEST in ("all", "unit", "unitandintegration")
        include("unit/unit_tests.jl")
    end

    if TRIXIPARTICLES_TEST in ("all", "integration", "unitandintegration")
        include("integration/integration_tests.jl")
    end

    if TRIXIPARTICLES_TEST in ("all", "system")
        include("system/system_tests.jl")
    end
end
