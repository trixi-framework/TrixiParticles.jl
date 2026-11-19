@testset verbose=true "DEMSystem (HertzContactModel)" begin
    @trixi_testset "show" begin
        # Define a simple 2D initial condition.
        coordinates = [1.0 2.0;
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, mass, density)

        # Create a Hertz contact model (only elastic modulus and Poisson's ratio).
        contact_model = HertzContactModel(1.0e10, 0.3)

        # Construct the DEM system.
        system = DEMSystem(initial_condition; contact_model, acceleration=(0.0, 10.0))

        # Expected compact representation.
        show_compact = "DEMSystem{2}(InitialCondition{Float64, Float64}(), HertzContactModel: elastic_modulus = 1.0e10, poissons_ratio = 0.3, damping_coefficient = 0.0001) with 2 particles"
        @test repr(system) == show_compact

        # Expected full text/plain representation.
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ DEMSystem{2}                                                                                     │
        │ ════════════                                                                                     │
        │ #particles: ………………………………………………… 2                                                                │
        │ elastic_modulus: …………………………………… 1.0e10                                                           │
        │ poissons_ratio: ……………………………………… 0.3                                                              │
        │ damping_coefficient: ………………………… 0.0001                                                           │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end

@testset verbose=true "DEMSystem (LinearContactModel)" begin
    @trixi_testset "show" begin
        # Define the same 2D initial condition.
        coordinates = [1.0 2.0;
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, mass, density)

        # Create a Linear contact model (with a normal stiffness value).
        contact_model = LinearContactModel(200000.0)

        # Construct the DEM system.
        system = DEMSystem(initial_condition; contact_model, acceleration=(0.0, 10.0))

        # Expected compact representation.
        show_compact = "DEMSystem{2}(InitialCondition{Float64, Float64}(), LinearContactModel: normal_stiffness = 200000.0, damping_coefficient = 0.0001) with 2 particles"
        @test repr(system) == show_compact

        # Expected full text/plain representation.
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ DEMSystem{2}                                                                                     │
        │ ════════════                                                                                     │
        │ #particles: ………………………………………………… 2                                                                │
        │ normal_stiffness: ………………………………… 200000.0                                                         │
        │ damping_coefficient: ………………………… 0.0001                                                           │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end

# Define a minimal dummy system to mimic DEMSystem for contact model tests.
struct DummySystem
    mass::Vector{Float64}    # mass per particle
    radius::Vector{Float64}  # radius per particle
end

# Define ndims for DummySystem so that any call to ndims returns 2 (for 2D systems).
Base.ndims(::DummySystem) = 2

@testset "DEM near-zero pair symmetry" begin
    ic = InitialCondition(; coordinates=zeros(Float32, 2, 1), mass=Float32[1],
                          density=1.0f0, particle_spacing=1.0f0)
    model = LinearContactModel(1.0f0)
    system_a = DEMSystem(ic; contact_model=model, radius=0.3f0,
                         damping_coefficient=0.0f0, acceleration=(0.0f0, 0.0f0))
    system_b = DEMSystem(ic; contact_model=model, radius=1.0f0,
                         damping_coefficient=0.0f0, acceleration=(0.0f0, 0.0f0))
    v = zeros(Float32, 2, 1)
    u_a, u_b = ic.coordinates, copy(ic.coordinates)
    semi = DummySemidiscretization()

    # The first separation is between the old cutoffs; the second retains the contact.
    for factor in (0.75f0, 1.5f0)
        u_b[1, 1] = factor * sqrt(eps(Float32))
        dv_a, dv_b = zero(v), zero(v)
        TrixiParticles.interact!(dv_a, v, u_a, v, u_b,
                                 system_a, system_b, semi)
        TrixiParticles.interact!(dv_b, v, u_b, v, u_a,
                                 system_b, system_a, semi)
        @test system_a.mass[1] * dv_a ≈ -system_b.mass[1] * dv_b
        @test iszero(dv_a) == (factor < 1)
    end
end

@testset "ContactModels Physical Behavior" begin

    # === HertzContactModel Tests ===
    @testset "HertzContactModel" begin
        # Material and contact parameters
        E = 1e10         # Elastic modulus
        nu = 0.3          # Poisson's ratio
        model = HertzContactModel(E, nu)

        # Geometric and contact configuration
        overlap = 0.1
        normal = SVector(1.0, 0.0)
        damping_coeff = 0.0001

        # Define a dummy particle system with one particle.
        mass = [2.0]      # [kg]
        radius = [0.5]      # [m]
        sysA = DummySystem(mass, radius)
        sysB = DummySystem(mass, radius)

        # -- Test 1: Zero relative velocity (pure elastic force) --
        vA = reshape([0.0, 0.0], (2, 1))  # 2×1 array for one particle (zero velocity)
        vB = reshape([0.0, 0.0], (2, 1))
        # For Hertz:
        # Effective modulus: E_star = 1 / ( ((1 - nu^2)/E + (1 - nu^2)/E) )
        E_star = 1 / (2 * ((1 - nu^2) / E))
        # Effective radius: r_star = (r_A * r_B)/(r_A + r_B)
        r_star = (radius[1] * radius[1]) / (radius[1] + radius[1])
        # Non-linear stiffness: (4/3)*E_star*sqrt(r_star*overlap)
        normal_stiffness = (4 / 3) * E_star * sqrt(r_star * overlap)
        elastic_force = normal_stiffness * overlap
        expected_force = elastic_force * normal

        computed_force = TrixiParticles.collision_force_normal(model, sysA, sysB, overlap,
                                                               normal, vA, vB, 1, 1,
                                                               damping_coeff)
        @test isapprox(computed_force, expected_force; rtol=1e-6)

        # -- Test 2: Nonzero relative velocity (elastic + damping contributions) --
        vA = reshape([1.0, 0.0], (2, 1))  # Particle A moving along x (nonzero velocity)
        vB = reshape([0.0, 0.0], (2, 1))
        rel_vel_norm = dot(SVector(1.0, 0.0) - SVector(0.0, 0.0), normal)  # equals 1.0
        # Effective mass: m_star = (m_A * m_B)/(m_A + m_B)
        m_star = (mass[1] * mass[1]) / (mass[1] + mass[1])
        # Critical damping coefficient: gamma_c = 2 * sqrt(m_star * normal_stiffness)
        gamma_c = 2 * sqrt(m_star * normal_stiffness)
        damping_term = damping_coeff * gamma_c * rel_vel_norm

        expected_force = (elastic_force + damping_term) * normal
        computed_force = TrixiParticles.collision_force_normal(model, sysA, sysB, overlap,
                                                               normal, vA, vB, 1, 1,
                                                               damping_coeff)
        @test isapprox(computed_force, expected_force; rtol=1e-6)
    end

    # === LinearContactModel Tests ===
    @testset "LinearContactModel" begin
        # Define a constant stiffness for the linear model.
        stiffness = 200000.0
        model = LinearContactModel(stiffness)

        # Contact configuration
        overlap = 0.1
        normal = SVector(1.0, 0.0)
        damping_coeff = 0.0001

        mass = [2.0]
        radius = [0.5]
        sysA = DummySystem(mass, radius)
        sysB = DummySystem(mass, radius)

        # -- Test 1: Zero relative velocity --
        vA = reshape([0.0, 0.0], (2, 1))
        vB = reshape([0.0, 0.0], (2, 1))
        elastic_force = stiffness * overlap
        expected_force = elastic_force * normal
        computed_force = TrixiParticles.collision_force_normal(model, sysA, sysB, overlap,
                                                               normal, vA, vB, 1, 1,
                                                               damping_coeff)
        @test isapprox(computed_force, expected_force; rtol=1e-6)

        # -- Test 2: Nonzero relative velocity --
        vA = reshape([1.0, 0.0], (2, 1))
        vB = reshape([0.0, 0.0], (2, 1))
        m_star = mass[1]
        gamma_c = 2 * sqrt(m_star * stiffness)
        damping_term = damping_coeff * gamma_c * 1.0
        expected_force = (stiffness * overlap + damping_term) * normal
        computed_force = TrixiParticles.collision_force_normal(model, sysA, sysB, overlap,
                                                               normal, vA, vB, 1, 1,
                                                               damping_coeff)
        @test isapprox(computed_force, expected_force; rtol=1e-6)
    end
end
