using Test
using TrixiParticles
using LinearAlgebra
using Printf

include("test_util.jl")

const TRIXI_TEST = lowercase(get(ENV, "TRIXI_TEST", "all"))

@testset "All Tests" verbose=true begin
    if TRIXI_TEST in ("all", "unit", "unitandintegration")
        include("unit/unit_tests.jl")
    end

    if TRIXI_TEST in ("all", "integration", "unitandintegration")
        include("integration/integration_tests.jl")
    end

    if TRIXI_TEST in ("all", "system")
        include("system/system_tests.jl")
    end
end
