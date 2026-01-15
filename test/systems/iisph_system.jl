@testset verbose=true "ImplicitIncompressibleSPHSystem" begin
    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors" begin
        # Verify constructor wiring, default values, and argument validation across 2D/3D
        coordinates_ = [
            [1.0 2.0
             1.0 2.0],
            [1.0 2.0
             1.0 2.0
             1.0 2.0]
        ]
        omegas = [
            0.5,
            0.6
        ]
        max_errors = [
            0.1,
            0.05
        ]
        min_iterations_ = [
            2,
            10
        ]
        max_iterations_ = [
            20,
            30
        ]
        time_steps_ = [
            0.001,
            0.0001
        ]
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]
            mass = [1.25, 1.5]
            density = [990.0, 1000.0]
            reference_density = 1000.0
            smoothing_kernel = Val(:smoothing_kernel)
            omega = omegas[i]
            max_error = max_errors[i]
            min_iterations = min_iterations_[i]
            max_iterations = max_iterations_[i]
            time_step = time_steps_[i]
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = i + 1
            smoothing_kernel2 = Val(:smoothing_kernel2)
            # The wrong dimension. 2 -> 3, 3 -> 2.
            TrixiParticles.ndims(::Val{:smoothing_kernel2}) = i % 2 + 2
            smoothing_length = 0.362

            initial_condition = InitialCondition(; coordinates, mass, density)
            system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density,
                                                     omega=omega,
                                                     max_error=max_error,
                                                     min_iterations=min_iterations,
                                                     max_iterations=max_iterations,
                                                     time_step=time_step)

            # Constructor copies input fields, applies defaults, and respects the requested dimensionality
            @test system isa ImplicitIncompressibleSPHSystem{NDIMS}
            @test system.initial_condition == initial_condition
            @test system.mass == mass
            @test system.reference_density == reference_density
            @test system.smoothing_kernel == smoothing_kernel
            @test system.omega == omega
            @test TrixiParticles.initial_smoothing_length(system) == smoothing_length
            @test system.viscosity === nothing
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test system.max_error == max_error
            @test system.min_iterations == min_iterations
            @test system.max_iterations == max_iterations
            @test system.time_step == time_step
            @test length(system.density) == size(coordinates, 2)

            # A too-short acceleration vector triggers dimension validation
            error_str1 = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str1) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   density,
                                                                                   acceleration=(0.0),
                                                                                   time_step=0.001)

            # Smoothing kernel dimensionality must match the problem dimension
            error_str2 = "smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str2) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel2,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   time_step=0.001)

            # Reference density must be positive
            error_str3 = "`reference_density` must be a positive number"
            @test_throws ArgumentError(error_str3) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   0.0,
                                                                                   time_step=0.001)

            # max_error is a percentage and must be in (0, 100]
            error_str4 = "`max_error` is given in percentage, so it must be a number between 0 and 100"
            @test_throws ArgumentError(error_str4) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   max_error=0.0,
                                                                                   time_step=0.001)

            # min_iterations must be strictly positive
            error_str5 = "`min_iterations` must be a positive number"
            @test_throws ArgumentError(error_str5) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=0,
                                                                                   time_step=0.001)

            # min_iterations must not exceed max_iterations
            error_str6 = "`min_iterations` must be smaller or equal to `max_iterations`"
            @test_throws ArgumentError(error_str6) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=10,
                                                                                   max_iterations=5,
                                                                                   time_step=0.001)

            # time_step must be strictly positive
            error_str7 = "`time_step must be a positive number"
            @test_throws ArgumentError(error_str6) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=10,
                                                                                   max_iterations=5,
                                                                                   time_step=0)
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors with Setups" begin
        # Validate setup-based construction for several geometries and ensure validation hooks persist
        setups = [
            RectangularShape(0.123, (2, 3), (-1.0, 0.1), density=1.0),
            RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), density=1.0),
            RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0).fluid,
            RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492),
                            1020.0).fluid,
            SphereShape(0.52, 0.1, (-0.2, 0.123), 1.0)
        ]
        setup_names = [
            "RectangularShape 2D",
            "RectangularShape 3D",
            "RectangularTank 2D",
            "RectangularTank 3D",
            "SphereShape 2D"
        ]
        NDIMS_ = [2, 3, 2, 3, 2]
        reference_densities = [1.0, 1.0, 1020.0, 1020.0, 1.0]
        @testset "$(setup_names[i])" for i in eachindex(setups)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            smoothing_kernel = Val(:smoothing_kernel)
            reference_density = reference_densities[i]
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362

            density_calculator = SummationDensity()

            system = ImplicitIncompressibleSPHSystem(setup,
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density,
                                                     time_step=0.001)

            # Constructor from setup copies geometry properties, sets defaults, and preserves dimension
            @test system isa ImplicitIncompressibleSPHSystem{NDIMS}
            @test system.initial_condition == setup
            @test system.mass == setup.mass
            @test system.smoothing_kernel == smoothing_kernel
            @test system.reference_density == reference_density
            @test TrixiParticles.initial_smoothing_length(system) == smoothing_length
            @test system.viscosity === nothing
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test length(system.mass) == size(setup.coordinates, 2)
            @test length(system.density) == size(setup.coordinates, 2)
        end

        # wrong dimension of acceleration
        NDIMS_ = [2, 3]
        @testset "Wrong acceleration dimension" for i in eachindex(NDIMS_)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            reference_density = 1.0
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362

            density_calculator = SummationDensity()

            # Acceleration vector length validation also applies to setup-based constructors
            error_str = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str) ImplicitIncompressibleSPHSystem(setup,
                                                                                  smoothing_kernel,
                                                                                  smoothing_length,
                                                                                  reference_density,
                                                                                  acceleration=(0.0),
                                                                                  time_step=0.001)
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 reference_density,
                                                 time_step=0.001)

        # repr without MIME should show compact one-line summary.
        show_compact = "ImplicitIncompressibleSPHSystem{2}(1000.0, Val{:smoothing_kernel}(), nothing, [0.0, 0.0], 0.5, 0.1, 2, 20) with 2 particles"
        @test repr(system) == show_compact
        # repr("text/plain", ...) should emit the detailed boxed summary.
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ ImplicitIncompressibleSPHSystem{2}                                                               │
        │ ══════════════════════════════════                                                               │
        │ #particles: ………………………………………………… 2                                                                │
        │ reference density: ……………………………… 1000.0                                                           │
        │ density calculator: …………………………… SummationDensity                                                 │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ viscosity: …………………………………………………… nothing                                                          │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        │ omega: ……………………………………………………………… 0.5                                                              │
        │ max_error: …………………………………………………… 0.1                                                              │
        │ min_iterations: ……………………………………… 2                                                                │
        │ max_iterations: ……………………………………… 20                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end

    @testset verbose=true "write_u0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                 smoothing_kernel,
                                                 smoothing_length, reference_density,
                                                 time_step=0.001)

        # `write_u0!` must copy initial coordinates into the solver state array
        u0 = zeros(TrixiParticles.u_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_u0!(u0, system)

        @test u0 == coordinates
    end

    @testset verbose=true "write_v0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        velocity = 2 * coordinates
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)

        # SummationDensity (is always in use)
        system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 reference_density,
                                                 time_step=0.001)

        # write_v0! must copy initial velocities into the solver state array
        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == velocity
    end

    # Validate helper routines that build the IISPH linear system and update pressures
    @trixi_testset "Solver helper calculations" begin
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.5

        coordinates = zeros(2, 1)
        velocity = zeros(2, 1)

        @testset "Fluid-fluid coefficients (d_ij, d_ii, sum term)" begin
            mass_a = [2.0]
            density_a = [5.0]
            pressure_a = [0.8]
            ic_a = InitialCondition(; coordinates, velocity, mass=mass_a, density=density_a,
                                    pressure=pressure_a)
            system = ImplicitIncompressibleSPHSystem(ic_a, smoothing_kernel,
                                                     smoothing_length, 6.0,
                                                     time_step=0.5)

            mass_b = [3.0]
            density_b = [4.0]
            pressure_b = [1.2]
            ic_b = InitialCondition(; coordinates, velocity, mass=mass_b, density=density_b,
                                    pressure=pressure_b)
            neighbor_system = ImplicitIncompressibleSPHSystem(ic_b, smoothing_kernel,
                                                              smoothing_length, 5.0,
                                                              time_step=0.5)

            system.sum_d_ij_pj[:, 1] .= (0.4, -0.2)
            neighbor_system.sum_d_ij_pj[:, 1] .= (0.05, 0.07)
            neighbor_system.d_ii[:, 1] .= (0.3, 0.1)

            grad_kernel = SVector(1.0, -0.5)

            # d_ij uses neighbor mass/density with kernel gradient sign; sum_term combines
            # both particles’ contributions per Ihmsen eq. 13.
            # With dt=0.5, m_b=3, rho_b=4, and grad_kernel=(1,-0.5), d_ij should be
            # (-0.5^2 * 3 / 4^2)*grad_kernel = (-0.046875, 0.0234375)
            @test isapprox(TrixiParticles.calculate_d_ij(system, neighbor_system, 1,
                                                         grad_kernel, system.time_step),
                           SVector(-0.046875, 0.0234375))
            # sum_term expands to m_j * dot(sum_dik_pk - d_jj * p_j - (sum_djk_pk - d_ji * p_i), grad_kernel)
            # with the prepared values this yields 3 * dot((0.4,-0.2) - (0.3,0.1)*1.2 - ((0.05,0.07) -
            # (-0.046875,0.0234375)*0.8), (1,-0.5)) = 0.615
            @test isapprox(TrixiParticles.calculate_sum_term(system, neighbor_system, 1,
                                                             1, grad_kernel,
                                                             system.time_step),
                           0.615; atol=1.0e-12)
            # d_ii uses particle i density and neighbor mass (eq. 9)
            # Using m_b=3, rho_a=5, grad_kernel[1]=1, dt=0.5 gives -dt^2 * m_b / rho_a^2 * grad = -0.03
            @test isapprox(TrixiParticles.calculate_d_ii(neighbor_system, mass_b[1],
                                                         density_a[1], grad_kernel[1],
                                                         system.time_step),
                           -0.03)
        end

        @testset "Boundary coefficients (PressureMirroring doubles a_ii)" begin
            mass_a = [2.0]
            density_a = [5.0]
            pressure_a = [0.8]
            ic_a = InitialCondition(; coordinates, velocity, mass=mass_a, density=density_a,
                                    pressure=pressure_a)
            system = ImplicitIncompressibleSPHSystem(ic_a, smoothing_kernel,
                                                     smoothing_length, 6.0,
                                                     time_step=0.5)
            system.sum_d_ij_pj[:, 1] .= (0.4, -0.2)

            grad_kernel = SVector(1.0, -0.5)

            struct MockBoundaryModel{DC}
                density_calculator::DC
            end
            struct MockBoundarySystem <: TrixiParticles.AbstractBoundarySystem{2}
                boundary_model :: MockBoundaryModel
                mass           :: Vector{Float64}
                density        :: Vector{Float64}
            end
            Base.ndims(::MockBoundarySystem) = 2
            TrixiParticles.hydrodynamic_mass(system::MockBoundarySystem,
                                             particle) = system.mass[particle]
            TrixiParticles.current_density(::Nothing, system::MockBoundarySystem,
                                           particle) = system.density[particle]
            TrixiParticles.current_density(_, system::MockBoundarySystem,
                                           particle) = system.density[particle]
            TrixiParticles.nparticles(system::MockBoundarySystem) = length(system.mass)

            boundary_system_mirroring = MockBoundarySystem(MockBoundaryModel(PressureMirroring()),
                                                           [1.5], [7.0])
            boundary_system_regular = MockBoundarySystem(MockBoundaryModel(AdamiPressureExtrapolation()),
                                                         [1.5], [7.0])

            # PressureMirroring removes off-diagonal pressure coupling and doubles diagonal term.
            # For the mirroring path: -dt^2 * m_b / rho_b^2 * grad = -0.25^2 * 1.5 / 7^2 * (1,-0.5)
            @test isapprox(TrixiParticles.calculate_d_ij(system, boundary_system_mirroring,
                                                         1,
                                                         grad_kernel, system.time_step),
                           SVector(-0.007653061224489796, 0.003826530612244898))
            regular_d_ii = TrixiParticles.calculate_d_ii(boundary_system_regular,
                                                         boundary_system_regular.boundary_model,
                                                         boundary_system_regular.boundary_model.density_calculator,
                                                         boundary_system_regular.mass[1],
                                                         density_a[1], grad_kernel[2],
                                                         system.time_step)
            # Regular boundary path (no mirroring) uses -dt^2 * m_b / rho_a^2 * grad_kernel[2] = 0.0075
            @test isapprox(regular_d_ii, 0.0075)
            @test isapprox(TrixiParticles.calculate_d_ii(boundary_system_mirroring,
                                                         boundary_system_mirroring.boundary_model,
                                                         boundary_system_mirroring.boundary_model.density_calculator,
                                                         boundary_system_mirroring.mass[1],
                                                         density_a[1], grad_kernel[2],
                                                         system.time_step),
                           2 * regular_d_ii; atol=1.0e-12)
            # Boundary sum_term should still accumulate the fluid-side sum_dij_pj contribution
            @test isapprox(TrixiParticles.calculate_sum_term(system,
                                                             boundary_system_mirroring, 1,
                                                             1, grad_kernel,
                                                             system.time_step),
                           0.75; atol=1.0e-12)
        end

        @testset "pressure_update handles zero diagonal entries" begin
            mass = [1.0, 1.0]
            density = [1000.0, 950.0]
            pressure = [0.0, 10.0]
            coordinates = [0.0 0.1
                           0.0 0.2]
            velocity = zeros(2, 2)
            ic = InitialCondition(; coordinates, velocity, mass, density, pressure)
            system_pressure = ImplicitIncompressibleSPHSystem(ic, smoothing_kernel,
                                                              smoothing_length, 1000.0,
                                                              omega=0.4, time_step=0.5)
            system_pressure.predicted_density .= [990.0, 1010.0]
            system_pressure.sum_term .= [5.0, -2.0]
            system_pressure.a_ii .= [0.5, 1.0e-10]
            fill!(system_pressure.density_error, 0.0)

            semi = DummySemidiscretization()
            # First particle uses standard Jacobi update; second hits the safeguarded zero-a_ii path.
            # For particle 1: (1-omega)*0 + omega/a_ii * (source - sum_term) with omega=0.4,
            # source=(1000-990)=10, a_ii=0.5, sum_term=5 gives pressure 4 and density_error -3
            relative_error = TrixiParticles.pressure_update(system_pressure,
                                                            system_pressure.pressure,
                                                            system_pressure.reference_density,
                                                            system_pressure.a_ii,
                                                            system_pressure.sum_term,
                                                            system_pressure.omega,
                                                            system_pressure.density_error,
                                                            semi)

            @test isapprox(relative_error, -0.003)
            @test isapprox(system_pressure.pressure, [4.0, 0.0])
            @test isapprox(system_pressure.density_error, [-3.0, 0.0])
        end

        @testset "Source term and iteration limits" begin
            mass = [1.0, 1.0]
            density = [1000.0, 950.0]
            pressure = [0.0, 0.0]
            coordinates = [0.0 0.1
                           0.0 0.2]
            velocity = zeros(2, 2)
            ic = InitialCondition(; coordinates, velocity, mass, density, pressure)
            system_iters = ImplicitIncompressibleSPHSystem(ic, smoothing_kernel,
                                                           smoothing_length, 1000.0,
                                                           omega=0.6, max_error=0.25,
                                                           min_iterations=3,
                                                           max_iterations=7,
                                                           time_step=0.25)
            system_iters.predicted_density .= [1005.0, 990.0]

            # Source term equals reference density minus predicted density for each particle
            # Here: 1000-1005 = -5 and 1000-990 = 10, matching the solver source definition
            @test TrixiParticles.iisph_source_term(system_iters, 1) == -5.0
            @test TrixiParticles.iisph_source_term(system_iters, 2) == 10.0
            # Accessors expose IISPH-specific particle counts and solver thresholds
            @test TrixiParticles.n_iisph_particles(system_iters) == 2
            @test TrixiParticles.maximum_iisph_error(system_iters) == 0.25
            @test TrixiParticles.minimum_iisph_iterations(system_iters) == 3
            @test TrixiParticles.maximum_iisph_iterations(system_iters) == 7
        end
    end
end
