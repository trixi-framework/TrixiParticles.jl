@trixi_testset "Correction Consistency" begin
    include("corrections/common.jl")
    include("corrections/lifecycle.jl")
    include("corrections/shepard.jl")
    include("corrections/kernel.jl")
    include("corrections/gradient.jl")
    include("corrections/mixed.jl")
    include("corrections/coupling.jl")
    include("corrections/configuration.jl")
end
