@trixi_testset "Correction Consistency" begin
    include("corrections/common.jl")
    include("corrections/lifecycle.jl")
    include("corrections/shepard.jl")
    include("corrections/kernel.jl")
    include("corrections/gradient.jl")
end
