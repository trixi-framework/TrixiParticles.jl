# Create a platform below the fluid (at a distance `walldistance`)
function create_boundary_system(coordinates, particle_spacing, state_equation, kernel,
                                smoothing_length, NDIMS, walldistance)
    # Compute bounding box of fluid particles
    xmin = minimum(coordinates[1, :])
    xmax = maximum(coordinates[1, :])
    ymin = minimum(coordinates[2, :])
    ymax = maximum(coordinates[2, :])

    wall_thickness = 4 * particle_spacing

    if NDIMS == 2
        wall_width = xmax - xmin
        wall_size = (wall_width, wall_thickness)
        wall_coord = (xmin, ymin - walldistance)
    elseif NDIMS == 3
        zmin = minimum(coordinates[3, :])
        wall_width_x = xmax - xmin
        wall_width_y = ymax - ymin
        wall_size = (wall_width_x, wall_width_y, wall_thickness)
        wall_coord = (xmin, ymin, zmin - walldistance)
    end

    # Create the wall shape
    wall = RectangularShape(particle_spacing,
                            round.(Int, wall_size ./ particle_spacing),
                            wall_coord,
                            density=1000.0)

    boundary_model = BoundaryModelDummyParticles(wall.density,
                                                 wall.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 kernel,
                                                 smoothing_length,
                                                 correction=nothing,
                                                 reference_particle_spacing=particle_spacing)

    boundary_system = BoundarySPHSystem(wall, boundary_model, adhesion_coefficient=0.0)
    return boundary_system
end

function create_fluid_system(coordinates, velocity, mass, density, particle_spacing;
                             NDIMS=2, smoothing_length=1.0, wall=false, walldistance=0.0)
    tspan = (0.0, 0.01)

    fluid = InitialCondition(coordinates=coordinates,
                             velocity=velocity,
                             mass=mass,
                             density=density,
                             particle_spacing=particle_spacing)

    state_equation = StateEquationCole(sound_speed=10.0,
                                       reference_density=1000.0,
                                       exponent=1)

    kernel = SchoenbergCubicSplineKernel{NDIMS}()

    system = WeaklyCompressibleSPHSystem(fluid,
                                         SummationDensity(),
                                         state_equation,
                                         kernel,
                                         smoothing_length;
                                         surface_normal_method=ColorfieldSurfaceNormal(),
                                         reference_particle_spacing=particle_spacing)

    if wall
        boundary_system = create_boundary_system(coordinates, particle_spacing,
                                                 state_equation, kernel, smoothing_length,
                                                 NDIMS, walldistance)
        semi = Semidiscretization(system, boundary_system)
    else
        semi = Semidiscretization(system)
        boundary_system = nothing
    end

    ode = semidiscretize(semi, tspan)
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    return system, boundary_system, semi, ode
end

function compute_and_test_surface_normals(system, semi, ode; NDIMS=2)
    surface_tension = SurfaceTensionAkinci()

    v0_ode, u0_ode = ode.u0.x
    v = TrixiParticles.wrap_v(v0_ode, system, semi)
    u = TrixiParticles.wrap_u(u0_ode, system, semi)

    TrixiParticles.compute_surface_normal!(system, surface_tension, v, u, v0_ode, u0_ode,
                                           semi, 0.0)

    # After computation, check that surface normals have been computed and are not NaN or Inf
    @test all(isfinite, system.cache.surface_normal)
    @test all(isfinite, system.cache.neighbor_count)
    @test size(system.cache.surface_normal, 1) == NDIMS

    nparticles = size(u, 2)

    # Check that the threshold has been applied correctly
    threshold = 2^ndims(system) + 1

    # Test the surface normals based on neighbor counts.
    # Test that surface normals are zero when there are not enough neighbors.
    # For the linear arrangement, surface normals may still be zero
    # when we have more neighbors than the threshold.
    @test all(i -> system.cache.neighbor_count[i] >= threshold ||
                  iszero(system.cache.surface_normal[:, i]), 1:nparticles)
end

@testset "Sphere Surface Normals" begin
    # Test case 1: Particles arranged in a disk
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
    system, bnd_system, semi, ode = create_fluid_system(coordinates, velocity, mass,
                                                        density,
                                                        particle_spacing;
                                                        NDIMS=NDIMS,
                                                        smoothing_length=3.0 *
                                                                         particle_spacing)

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
    @test all(isapprox.(computed_normals[:, surface_particles],
                        expected_normals[:, surface_particles], atol=0.05))

    # Optionally, check that normals for interior particles are zero.
    # This will always trigger since the outer layer which has normals set is usually 2-3 particles thick.
    # The thickness of the normal layers cannot be controlled with the method currently implemented.
    # for i in setdiff(1:nparticles, surface_particles)
    #     @test isapprox(norm(system.cache.surface_normal[:, i]), 0.0, atol=1e-4)
    # end
end

@testset "Sphere Surface Normals with Wall" begin
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
    system, bnd_system, semi, ode = create_fluid_system(coordinates, velocity, mass,
                                                        density,
                                                        particle_spacing;
                                                        NDIMS=NDIMS,
                                                        smoothing_length=3.0 *
                                                                         particle_spacing,
                                                        wall=true,
                                                        walldistance=2.0)

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

    # Boundary system
    bnd_color = bnd_system.boundary_model.cache.colorfield_bnd
    # This is only true since it assumed that the color is 1
    @test all(bnd_color .>= 0.0)

    # Compare computed normals to expected normals for surface particles (with wall we have a larger error)
    @test all(isapprox.(computed_normals[:, surface_particles],
                        expected_normals[:, surface_particles], atol=0.5))

    # Optionally, check that normals for interior particles are zero.
    # This will always trigger since the outer layer which has normals set is usually 2-3 particles thick.
    # The thickness of the normal layers cannot be controlled with the method currently implemented.
    # for i in setdiff(1:nparticles, surface_particles)
    #     @test isapprox(norm(system.cache.surface_normal[:, i]), 0.0, atol=1e-4)
    # end
end
