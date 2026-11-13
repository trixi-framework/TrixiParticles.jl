@testset verbose=true "RigidBodySystem" begin
    include("rigid_body/core.jl")
    include("rigid_body/state_io.jl")
    include("rigid_body/fluid_interaction.jl")
    include("rigid_body/dynamics.jl")
    include("rigid_body/normal_contact.jl")
    include("rigid_body/contact_model.jl")
    include("rigid_body/contact_history.jl")
end
