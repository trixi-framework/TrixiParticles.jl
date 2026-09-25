# Different references guard different failure modes: the core exercises local numerical
# primitives, recorded values catch changes to the production pipeline, the API tests
# stateful use, and the independent checks compare against analytic geometry.
include("core.jl")
include("reference_values.jl")
include("api.jl")
include("correctness.jl")
