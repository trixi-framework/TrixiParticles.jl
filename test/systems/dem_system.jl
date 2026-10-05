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
    for T in (Float32, Float64)
        ic = InitialCondition(; coordinates=zeros(T, 2, 1), mass=T[1],
                              density=one(T), particle_spacing=one(T))
        model = LinearContactModel(one(T))
        system_a = DEMSystem(ic; contact_model=model, radius=T(0.3),
                             damping_coefficient=zero(T), acceleration=(zero(T), zero(T)))
        system_b = DEMSystem(ic; contact_model=model, radius=one(T),
                             damping_coefficient=zero(T), acceleration=(zero(T), zero(T)))
        v = zeros(T, 2, 1)
        u_a, u_b = ic.coordinates, copy(ic.coordinates)
        semi = DummySemidiscretization()

        # Include coincidence, a separation between the old system-based cutoffs,
        # the exact strict-comparison boundary, and a larger retained separation.
        for factor in T.((0.0, 0.75, 1.0, 1.5))
            u_b[1, 1] = factor * sqrt(eps(T))
            dv_a, dv_b = zero(v), zero(v)
            TrixiParticles.interact!(dv_a, v, u_a, v, u_b,
                                     system_a, system_b, semi)
            TrixiParticles.interact!(dv_b, v, u_b, v, u_a,
                                     system_b, system_a, semi)
            @test system_a.mass[1] * dv_a ≈ -system_b.mass[1] * dv_b
            @test iszero(dv_a) == (factor < 1)
            @test iszero(dv_b) == (factor < 1)
            if factor >= 1
                expected_force = SVector(-(T(0.3) + one(T) - u_b[1, 1]), zero(T))
                @test system_a.mass[1] * dv_a[:, 1] ≈ expected_force
            end
        end

        # An unrelated large particle must not suppress the smaller pair's contact.
        ic_large = InitialCondition(; coordinates=T[0 1.0e5; 0 0], mass=ones(T, 2),
                                    density=one(T), particle_spacing=one(T))
        system_large = DEMSystem(ic_large; contact_model=model, radius=T(0.3),
                                 damping_coefficient=zero(T),
                                 acceleration=(zero(T), zero(T)))
        system_large.radius[2] = T(1000)
        u_b[1, 1] = T(0.75) * sqrt(eps(T)) * system_large.radius[2]
        v_large = zeros(T, 2, 2)
        dv_large, dv_b = zero(v_large), zero(v)
        TrixiParticles.interact!(dv_large, v_large, ic_large.coordinates, v, u_b,
                                 system_large, system_b, semi)
        TrixiParticles.interact!(dv_b, v, u_b, v_large, ic_large.coordinates,
                                 system_b, system_large, semi)
        @test !iszero(dv_large[:, 1])
        @test iszero(dv_large[:, 2])
        @test system_large.mass[1] * dv_large[:, 1] ≈ -system_b.mass[1] * dv_b[:, 1]
        expected_force = SVector(-(T(0.3) + one(T) - u_b[1, 1]), zero(T))
        @test system_large.mass[1] * dv_large[:, 1] ≈ expected_force
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
