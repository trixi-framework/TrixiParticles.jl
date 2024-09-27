include("../../test_util.jl")

@testset verbose=true "Surface Normal Computation" begin
    # Define necessary parameters
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.0
    ELTYPE = Float64
    NDIMS = 2
    nparticles = 5

    # Create an instance of ColorfieldSurfaceNormal
    surfn = ColorfieldSurfaceNormal(smoothing_kernel, smoothing_length)
    @test surfn.smoothing_kernel === smoothing_kernel
    @test surfn.smoothing_length == smoothing_length

    # Correctly create coordinates and velocity as matrices
    coordinates = zeros(NDIMS, nparticles)
    for i in 1:nparticles
        coordinates[:, i] = [i * 1.0, 0.0]  # Position of each particle
    end

    sound_speed = 20.0
    fluid_density = 1000.0
    velocity = zeros(NDIMS, nparticles)  # Zero velocity for all particles
    mass = fill(1.0, nparticles)
    density = fill(fluid_density, nparticles)
    pressure = fill(0.0, nparticles)

    # Create the InitialCondition with particle_spacing
    fluid = InitialCondition(coordinates=coordinates, velocity=velocity, mass=mass,
                             density=density, particle_spacing=1.0)

    # Create the FluidSystem, passing surfn as surface_normal_method
    state_equation = StateEquationCole(sound_speed=sound_speed, reference_density=fluid_density,
                                       exponent=1, clip_negative_pressure=false)
    density_calculator = SummationDensity()
    system = WeaklyCompressibleSPHSystem(fluid, density_calculator, state_equation,
                                         smoothing_kernel, smoothing_length;
                                         surface_normal_method=surfn,
                                         reference_particle_spacing=1.0,
                                         buffer_size=0)  # Set buffer_size to 0

    # Create the Semidiscretization
    semi = Semidiscretization(system)

    # Create the ODE problem
    tspan = (0.0, 0.01)
    ode = semidiscretize(semi, tspan)

    # Initialize the neighborhood search and update systems
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    # Extract u and v arrays for positions and velocities using wrap_u and wrap_v
    u = TrixiParticles.wrap_u(ode.u0, system, semi)
    v = TrixiParticles.wrap_v(ode.u0, system, semi)

    # Test compute_surface_normal! function directly
    surface_tension = SurfaceTensionAkinci()
    TrixiParticles.compute_surface_normal!(system, surface_tension, v, u, ode.u0, ode.u0, semi, 0.0)

    # After computation, check that surface normals have been computed
    @test all(isfinite.(system.cache.surface_normal))
    @test all(isfinite.(system.cache.neighbor_count))
    @test size(system.cache.surface_normal) == (NDIMS, nparticles)

    # Test remove_invalid_normals!
    # Manually set neighbor counts to test the removal of invalid normals
    system.cache.neighbor_count .= [3, 2, 1, 4, 0]
    TrixiParticles.remove_invalid_normals!(system, surface_tension)

    # Normals for particles with neighbor_count < threshold should be zero
    threshold = 2^NDIMS + 1
    for i in 1:nparticles
        if system.cache.neighbor_count[i] < threshold
            @test all(system.cache.surface_normal[:, i] .== 0.0)
        else
            @test any(system.cache.surface_normal[:, i] .!= 0.0)
        end
    end

    @testset verbose=true "Sphere Surface Normals" begin
        # Define necessary parameters
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        ELTYPE = Float64
        NDIMS = 2

        # Create an instance of ColorfieldSurfaceNormal
        surfn = ColorfieldSurfaceNormal(smoothing_kernel, smoothing_length)

        # Create the InitialCondition with particles arranged in a circle
        particle_spacing = 0.1
        radius = 1.0
        center = (0.0, 0.0)
        fluid_density = 1000.0

        # Create a SphereShape (which is a circle in 2D)
        sphere_ic = SphereShape(particle_spacing, radius, center, fluid_density)

        # Extract the number of particles
        nparticles = size(sphere_ic.coordinates, 2)

        # Create the FluidSystem
        sound_speed = 20.0
        velocity = zeros(NDIMS, nparticles)  # Zero velocity for all particles
        mass = sphere_ic.mass
        density = sphere_ic.density

        fluid = InitialCondition(coordinates=sphere_ic.coordinates, velocity=velocity,
                                 mass=mass, density=density, particle_spacing=particle_spacing)

        # Create the FluidSystem, passing surfn as surface_normal_method
        state_equation = StateEquationCole(sound_speed=sound_speed,
                                           reference_density=fluid_density,
                                           exponent=1, clip_negative_pressure=false)
        density_calculator = SummationDensity()
        system = WeaklyCompressibleSPHSystem(fluid, density_calculator, state_equation,
                                             smoothing_kernel, smoothing_length;
                                             surface_normal_method=surfn,
                                             reference_particle_spacing=particle_spacing,
                                             buffer_size=0)

        # Create the Semidiscretization
        semi = Semidiscretization(system)

        # Create the ODE problem
        tspan = (0.0, 0.01)
        ode = semidiscretize(semi, tspan)

        # Initialize the neighborhood search and update systems
        TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

        # Extract u and v arrays for positions and velocities using wrap_u and wrap_v
        u = TrixiParticles.wrap_u(ode.u0, system, semi)
        v = TrixiParticles.wrap_v(ode.u0, system, semi)

        # Compute the surface normals
        surface_tension = SurfaceTensionAkinci()
        TrixiParticles.compute_surface_normal!(system, surface_tension, v, u, ode.u0, ode.u0, semi, 0.0)

        # After computation, check that surface normals have been computed
        @test all(isfinite.(system.cache.surface_normal))
        @test all(isfinite.(system.cache.neighbor_count))
        @test size(system.cache.surface_normal) == (NDIMS, nparticles)

        # Define a small epsilon for surface detection
        epsilon = particle_spacing / 2

        # Initialize arrays to hold expected normals and indices of surface particles
        expected_normals = zeros(NDIMS, nparticles)
        surface_particles = Int[]

        # Compute expected normals and identify surface particles
        for i in 1:nparticles
            pos = u[:, i]
            r = pos - SVector(center...)
            norm_r = norm(r)
            if abs(norm_r - radius) < epsilon
                expected_normals[:, i] = r / norm_r
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
            @test isapprox(computed_normals[:, i], expected_normals[:, i])
        end

        # Optionally, check that normals for interior particles are zero
        for i in setdiff(1:nparticles, surface_particles)
            @test isapprox(norm(system.cache.surface_normal[:, i]), 0.0)
        end
    end
end
