@testset verbose=true "Source Terms" begin
    @testset verbose=true "SpongeLayer" begin
        # Outflow face at x = 2 with the fluid domain at x < 2
        sponge = SpongeLayer(; face_origin=(2.0, 0.0), face_normal=(-2.0, 0.0),
                             length=0.5, sound_speed=10.0,
                             reference_velocity=(1.0, 0.0))

        @testset "Constructor" begin
            @test sponge.face_normal == SVector(-1.0, 0.0)
            @test sponge.max_relaxation_rate ≈ 3 * 10.0 / 0.5
            @test repr(sponge) == "SpongeLayer{2, Float64}(length=0.5, max_relaxation_rate=60.0)"

            sponge2 = SpongeLayer(; face_origin=(2.0, 0.0), face_normal=(-1.0, 0.0),
                                  length=0.5, sound_speed=10.0, strength=0.5,
                                  reference_velocity=(1.0, 0.0))
            @test sponge2.max_relaxation_rate ≈ 0.5 * sponge.max_relaxation_rate

            error_str = "`length` must be positive"
            @test_throws ArgumentError(error_str) SpongeLayer(; face_origin=(2.0, 0.0),
                                                              face_normal=(-1.0, 0.0),
                                                              length=0.0,
                                                              sound_speed=10.0,
                                                              reference_velocity=(1.0,
                                                                                  0.0))

            error_str = "`face_normal` must have the same length as `face_origin`"
            @test_throws ArgumentError(error_str) SpongeLayer(; face_origin=(2.0, 0.0),
                                                              face_normal=(-1.0, 0.0,
                                                                           0.0),
                                                              length=0.5,
                                                              sound_speed=10.0,
                                                              reference_velocity=(1.0,
                                                                                  0.0))

            error_str = "`reference_velocity` must be a function or a vector of length 2"
            @test_throws ArgumentError(error_str) SpongeLayer(; face_origin=(2.0, 0.0),
                                                              face_normal=(-1.0, 0.0),
                                                              length=0.5,
                                                              sound_speed=10.0,
                                                              reference_velocity=1.0)
        end

        @testset "Evaluation" begin
            velocity = SVector(3.0, 1.0)
            density = 1000.0
            pressure = 0.0
            t = 0.0

            # Outside of the layer: upstream, and behind the boundary face
            for x in (1.4, 2.1)
                @test sponge(SVector(x, 0.3), velocity, density, pressure, t) ==
                      zero(velocity)
            end

            # At the boundary face, the full relaxation rate is applied
            @test sponge(SVector(2.0, 0.3), velocity, density, pressure, t) ≈
                  -60.0 * (velocity - SVector(1.0, 0.0))

            # Halfway into the layer, the relaxation rate is a quarter of the maximum
            @test sponge(SVector(1.75, 0.3), velocity, density, pressure, t) ≈
                  -15.0 * (velocity - SVector(1.0, 0.0))

            # No damping when the velocity equals the reference velocity
            @test sponge(SVector(1.9, 0.3), SVector(1.0, 0.0), density, pressure, t) ==
                  zero(velocity)

            # Reference velocity as a function of the coordinates and time
            v_ref(pos, t) = SVector(pos[2] * t, 0.0)
            sponge_func = SpongeLayer(; face_origin=(2.0, 0.0), face_normal=(-1.0, 0.0),
                                      length=0.5, sound_speed=10.0,
                                      reference_velocity=v_ref)
            @test sponge_func(SVector(2.0, 0.3), velocity, density, pressure, 2.0) ≈
                  -60.0 * (velocity - SVector(0.6, 0.0))
        end

        @testset "Non-reflecting Impedance" begin
            # The integral of the relaxation rate over the layer must equal
            # `strength * sound_speed` for a non-reflecting layer (see docstring).
            n = 10_000
            dx = sponge.length / n
            integral = sum(1:n) do i
                x = 2.0 - (i - 0.5) * dx
                # The velocity differs from the reference velocity by one,
                # so this is the relaxation rate at this distance from the face.
                -sponge(SVector(x, 0.0), SVector(2.0, 0.0), 0.0, 0.0, 0.0)[1]
            end * dx
            @test isapprox(integral, 10.0, rtol=1e-6)
        end

        @testset "BoundaryZone Constructor" begin
            outflow = BoundaryZone(; boundary_face=([2.0, 0.0], [2.0, 1.0]),
                                   particle_spacing=0.05, sample_points=nothing,
                                   face_normal=(-1.0, 0.0), density=1.0,
                                   reference_velocity=[1.0, 0.0],
                                   open_boundary_layers=4, boundary_type=OutFlow())

            sponge_zone = SpongeLayer(outflow; length=0.5, sound_speed=10.0)
            @test sponge_zone.face_origin == SVector(2.0, 0.0)
            @test sponge_zone.face_normal == SVector(-1.0, 0.0)
            @test sponge_zone.max_relaxation_rate ≈ sponge.max_relaxation_rate
            # The prescribed velocity of the boundary zone is used as reference velocity
            @test sponge_zone(SVector(2.0, 0.3), SVector(3.0, 1.0), 1.0, 0.0, 0.0) ≈
                  -60.0 * SVector(2.0, 1.0)

            # Passing a reference velocity overrides the one of the boundary zone
            sponge_zone2 = SpongeLayer(outflow; length=0.5, sound_speed=10.0,
                                       reference_velocity=(0.0, 0.0))
            @test sponge_zone2(SVector(2.0, 0.3), SVector(3.0, 1.0), 1.0, 0.0, 0.0) ≈
                  -60.0 * SVector(3.0, 1.0)

            outflow_pressure = BoundaryZone(; boundary_face=([2.0, 0.0], [2.0, 1.0]),
                                            particle_spacing=0.05, sample_points=nothing,
                                            face_normal=(-1.0, 0.0), density=1.0,
                                            reference_pressure=0.0,
                                            open_boundary_layers=4,
                                            boundary_type=OutFlow())

            error_str = "the boundary zone has no prescribed velocity, " *
                        "so `reference_velocity` must be passed"
            @test_throws ArgumentError(error_str) SpongeLayer(outflow_pressure;
                                                              length=0.5,
                                                              sound_speed=10.0)
        end
    end
end
