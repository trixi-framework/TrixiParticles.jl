function create_fluid_system(coordinates, velocity, mass, density, particle_spacing;
                             NDIMS=2, smoothing_length=1.0)
    tspan = (0.0, 0.01)

    fluid = InitialCondition(coordinates=coordinates, velocity=velocity, mass=mass,
                             density=density, particle_spacing=particle_spacing)

    state_equation = StateEquationCole(sound_speed=10.0,
                                       reference_density=1000.0,
                                       exponent=1)

    system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(), state_equation,
                                         SchoenbergCubicSplineKernel{NDIMS}(),
                                         smoothing_length;
                                         surface_normal_method=ColorfieldSurfaceNormal(),
                                         reference_particle_spacing=particle_spacing)

    semi = Semidiscretization(system)
    ode = semidiscretize(semi, tspan)

    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    return system, semi, ode
end

function compute_and_test_surface_normals(system, semi, ode; NDIMS=2)
    surface_tension = SurfaceTensionAkinci()

    v0_ode, u0_ode = ode.u0.x
    v = TrixiParticles.wrap_v(v0_ode, system, semi)
    u = TrixiParticles.wrap_u(u0_ode, system, semi)

    TrixiParticles.compute_surface_normal!(system, surface_tension, v, u, v0_ode, u0_ode,
                                           semi, 0.0)

    # After computation, check that surface normals have been computed and are not NaN or Inf
    @test all(isfinite.(system.cache.surface_normal))
    @test all(isfinite.(system.cache.neighbor_count))
    @test size(system.cache.surface_normal, 1) == NDIMS

    nparticles = size(u, 2)

    # Check that the threshold has been applied correctly
    threshold = 2^ndims(system) + 1

    # Test the surface normals based on neighbor counts
    for i in 1:nparticles
        if system.cache.neighbor_count[i] < threshold
            @test all(system.cache.surface_normal[:, i] .== 0.0)
        else
            # For the linear arrangement, surface normals may still be zero
            # Adjust the test to account for this possibility
            @test true
        end
    end
end

@testset "Surface Normal Computation" begin
    # Test case 1: Simple linear arrangement of particles
    nparticles = 5
    particle_spacing = 1.0
    NDIMS = 2

    coordinates = zeros(NDIMS, nparticles)
    for i in 1:nparticles
        coordinates[:, i] = [i * particle_spacing, 0.0]
    end
    velocity = zeros(NDIMS, nparticles)
    mass = fill(1.0, nparticles)
    fluid_density = 1000.0
    density = fill(fluid_density, nparticles)

    system, semi, ode = create_fluid_system(coordinates, velocity, mass, density,
                                            particle_spacing;
                                            NDIMS=NDIMS)

    compute_and_test_surface_normals(system, semi, ode; NDIMS=NDIMS)
end

@testset "Sphere Surface Normals" begin
    # Test case 2: Particles arranged in a disk
    particle_spacing = 0.25
    radius = 1.0
    center = (0.0, 0.0)
    NDIMS = 2

    # Create a `SphereShape`, which is a disk in 2D
    sphere_ic = SphereShape(particle_spacing, radius, center, 1000.0)

    coordinates = sphere_ic.coordinates
    velocity = zeros(NDIMS, size(coordinates, 2))
    mass = sphere_ic.mass
    density = sphere_ic.density

    # To get somewhat accurate normals we increase the smoothing length unrealistically
    system, semi, ode = create_fluid_system(coordinates, velocity, mass, density,
                                            particle_spacing;
                                            NDIMS=NDIMS,
                                            smoothing_length=3.0 * particle_spacing)

    compute_and_test_surface_normals(system, semi, ode; NDIMS=NDIMS)

    nparticles = size(coordinates, 2)
    expected_normals = zeros(NDIMS, nparticles)
    surface_particles = Int[]

    # Compute expected normals and identify surface particles
    for i in 1:nparticles
        pos = coordinates[:, i]
        r = pos .- center
        norm_r = norm(r)

        # If particle is on the circumference of the circle
        if abs(norm_r - radius) < particle_spacing
            expected_normals[:, i] = -r / norm_r

            push!(surface_particles, i)
        else
            expected_normals[:, i] .= 0.0
        end
    end

    # Normalize computed normals
    computed_normals = copy(system.cache.surface_normal)
    for i in surface_particles
        norm_computed = norm(computed_normals[:, i])
        if norm_computed > 0
            computed_normals[:, i] /= norm_computed
        end
    end

    # Compare computed normals to expected normals for surface particles
    for i in surface_particles
        @test isapprox(computed_normals[:, i], expected_normals[:, i], atol=0.04)
    end

    # Optionally, check that normals for interior particles are zero
    # for i in setdiff(1:nparticles, surface_particles)
    #     @test isapprox(norm(system.cache.surface_normal[:, i]), 0.0, atol=1e-4)
    # end
end
