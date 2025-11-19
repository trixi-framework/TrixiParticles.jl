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

    boundary_system = WallBoundarySystem(wall, boundary_model, adhesion_coefficient=0.0)
    return boundary_system
end

function create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                             surface_tension;
                             surface_normal_method=ColorfieldSurfaceNormal(),
                             NDIMS=2, smoothing_length=1.0, wall=false, walldistance=0.0,
                             smoothing_kernel=SchoenbergCubicSplineKernel{NDIMS}())
    tspan = (0.0, 0.01)

    fluid = InitialCondition(coordinates=coordinates,
                             velocity=velocity,
                             mass=mass,
                             density=density,
                             particle_spacing=particle_spacing)

    state_equation = StateEquationCole(sound_speed=10.0,
                                       reference_density=1000.0,
                                       exponent=1)

    system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(), state_equation,
                                         smoothing_kernel, smoothing_length;
                                         surface_normal_method=surface_normal_method,
                                         reference_particle_spacing=particle_spacing,
                                         surface_tension=surface_tension)

    if wall
        boundary_system = create_boundary_system(coordinates, particle_spacing,
                                                 state_equation, smoothing_kernel,
                                                 smoothing_length,
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

function compute_and_test_surface_values(system, semi, ode; NDIMS=2)
    v0_ode, u0_ode = ode.u0.x
    v = TrixiParticles.wrap_v(v0_ode, system, semi)
    u = TrixiParticles.wrap_u(u0_ode, system, semi)

    # Compute the surface normals
    TrixiParticles.compute_surface_normal!(system, system.surface_normal_method, v, u,
                                           v0_ode, u0_ode, semi, 0.0)

    TrixiParticles.remove_invalid_normals!(system, system.surface_tension,
                                           system.surface_normal_method)

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

function compute_curvature!(system, semi, ode)
    v0_ode, u0_ode = ode.u0.x
    v = TrixiParticles.wrap_v(v0_ode, system, semi)
    u = TrixiParticles.wrap_u(u0_ode, system, semi)

    TrixiParticles.compute_curvature!(system, system.surface_tension,
                                      v, u, v0_ode, u0_ode, semi, 0.0)
end

@testset verbose=true "CSS/CSF: Sphere Surface Normals" begin
    # Define each variation as a tuple of parameters:
    # (NDIMS, smoothing_kernel, particle_spacing, smoothing_length_multiplier, radius, center, relative_curvature_error)
    variations = [
        (2, SchoenbergCubicSplineKernel{2}(), 0.2, 3.0, 1.0, (0.0, 0.0), 0.8),
        (2, SchoenbergCubicSplineKernel{2}(), 0.1, 3.5, 1.0, (0.0, 0.0), 1.7),
        (3, SchoenbergCubicSplineKernel{3}(), 0.25, 3.0, 1.0, (0.0, 0.0, 0.0), 0.5),
        (2, WendlandC2Kernel{2}(), 0.3, 1.0, 1.0, (0.0, 0.0), 1.4),
        (3, WendlandC2Kernel{3}(), 0.3, 1.5, 1.0, (0.0, 0.0, 0.0), 0.6)
    ]

    for (NDIMS, smoothing_kernel, particle_spacing, smoothing_length_mult, radius, center,
         relative_curvature_error) in variations

        @testset "NDIMS: $(NDIMS), Kernel: $(typeof(smoothing_kernel)), spacing: $(particle_spacing)" begin
            smoothing_length = smoothing_length_mult * particle_spacing

            # Create a `SphereShape`, which is a disk in 2D
            sphere_ic = SphereShape(particle_spacing, radius, center, 1000.0)

            coordinates = sphere_ic.coordinates
            velocity = zeros(NDIMS, size(coordinates, 2))
            mass = sphere_ic.mass
            density = sphere_ic.density

            # wall is placed 2.0 away so that it doesn't have much influence on the result
            system, bnd_system, semi,
            ode = create_fluid_system(coordinates, velocity, mass,
                                      density,
                                      particle_spacing,
                                      SurfaceTensionMorris(surface_tension_coefficient=0.072);
                                      NDIMS=NDIMS,
                                      smoothing_length=smoothing_length,
                                      smoothing_kernel=smoothing_kernel,
                                      surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                                    ideal_density_threshold=0.9),
                                      wall=true, walldistance=2.0)

            compute_and_test_surface_values(system, semi, ode; NDIMS=NDIMS)

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
            bnd_color = bnd_system.boundary_model.cache.initial_colorfield
            # This is only true since it assumed that the color is 1
            @test all(bnd_color .>= 0.0)

            # Test that computed normals match expected normals
            @test isapprox(computed_normals[:, surface_particles],
                           expected_normals[:, surface_particles], norm=x -> norm(x, Inf),
                           atol=0.04)

            compute_curvature!(system, semi, ode)

            # Check that curvature is finite
            @test all(isfinite, system.cache.curvature)

            # Theoretical curvature magnitude
            #  - circle (2D):  1 / radius
            #  - sphere (3D):  2 / radius
            expected_curv = (NDIMS == 2) ? (1.0 / radius) : (2.0 / radius)
            curvature = system.cache.curvature

            # Compare absolute value of computed curvature vs. expected
            for i in surface_particles
                @test isapprox(abs(curvature[i]),
                               expected_curv;
                               atol=relative_curvature_error * expected_curv)
            end

            # Optionally, test that interior particles have near-zero normals
            # for i in setdiff(1:nparticles, surface_particles)
            #     @test isapprox(norm(system.cache.surface_normal[:, i]), 0.0, atol=1e-4)
            # end
        end
    end
end

@testset verbose=true "Akinci Sphere Surface Normals" begin
    # Define each variation as a tuple of parameters:
    # (NDIMS, smoothing_kernel, particle_spacing, smoothing_length_multiplier, radius, center, relative_curvature_error)
    variations = [
        (2, SchoenbergCubicSplineKernel{2}(), 0.2, 3.0, 1.0, (0.0, 0.0), 0.8),
        (2, SchoenbergCubicSplineKernel{2}(), 0.1, 3.5, 1.0, (0.0, 0.0), 1.7),
        (3, SchoenbergCubicSplineKernel{3}(), 0.25, 3.0, 1.0, (0.0, 0.0, 0.0), 0.5),
        (2, WendlandC2Kernel{2}(), 0.3, 1.0, 1.0, (0.0, 0.0), 1.4),
        (3, WendlandC2Kernel{3}(), 0.3, 1.5, 1.0, (0.0, 0.0, 0.0), 0.6)
    ]

    for (NDIMS, smoothing_kernel, particle_spacing, smoothing_length_mult, radius, center,
         relative_curvature_error) in variations

        @testset "NDIMS: $(NDIMS), Kernel: $(typeof(smoothing_kernel)), spacing: $(particle_spacing)" begin
            smoothing_length = smoothing_length_mult * particle_spacing

            # Create a `SphereShape`, which is a disk in 2D
            sphere_ic = SphereShape(particle_spacing, radius, center, 1000.0)

            coordinates = sphere_ic.coordinates
            velocity = zeros(NDIMS, size(coordinates, 2))
            mass = sphere_ic.mass
            density = sphere_ic.density

            system, bnd_system, semi,
            ode = create_fluid_system(coordinates, velocity, mass,
                                      density,
                                      particle_spacing,
                                      SurfaceTensionAkinci(surface_tension_coefficient=0.072);
                                      NDIMS=NDIMS,
                                      smoothing_length=smoothing_length,
                                      smoothing_kernel=smoothing_kernel,
                                      surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                                    ideal_density_threshold=0.9),
                                      wall=true, walldistance=2.0)

            compute_and_test_surface_values(system, semi, ode; NDIMS=NDIMS)

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
            bnd_color = bnd_system.boundary_model.cache.initial_colorfield
            # this is only true since it assumed that the color is 1
            @test all(bnd_color .>= 0.0)

            # Test that computed normals match expected normals
            @test isapprox(computed_normals[:, surface_particles],
                           expected_normals[:, surface_particles], norm=x -> norm(x, Inf),
                           atol=0.04)

            # Optionally, test that interior particles have near-zero normals
            # for i in setdiff(1:nparticles, surface_particles)
            #     @test isapprox(norm(system.cache.surface_normal[:, i]), 0.0, atol=1e-4)
            # end
        end
    end
end

@testset "Rectangular Fluid with Corner Normal Check" begin
    # Domain dimensions
    width = 2.0
    height = 1.0
    particle_spacing = 0.1
    NDIMS = 2

    # Generate a rectangular grid of coordinates from (0,0) to (width,height)
    x_vals = 0.0:particle_spacing:width
    y_vals = 0.0:particle_spacing:height

    coords_list = []
    for y in y_vals
        for x in x_vals
            push!(coords_list, [x, y])
        end
    end
    coordinates = hcat(coords_list...)   # size(coordinates) == (2, N)
    nparticles = size(coordinates, 2)

    # Initialize velocity, mass, density
    velocity = zeros(NDIMS, nparticles)
    mass = fill(1.0, nparticles)
    fluid_density = 1000.0
    density = fill(fluid_density, nparticles)

    # Create fluid system (no wall)
    system, bnd_system, semi,
    ode = create_fluid_system(coordinates, velocity, mass,
                              density, particle_spacing,
                              SurfaceTensionMorris(surface_tension_coefficient=0.072);
                              NDIMS=NDIMS,
                              smoothing_length=1.5 *
                                               particle_spacing,
                              wall=false, walldistance=0.0)

    # Compute surface normals
    compute_and_test_surface_values(system, semi, ode; NDIMS=NDIMS)

    # Threshold to decide if a particle is "on" a boundary
    # (half the spacing is typical, adjust as needed)
    surface_threshold = 0.5 * particle_spacing

    # Function to compute the "expected" outward normal of the rectangle
    function expected_rect_normal(pos, w, h, surface_threshold)
        x, y = pos
        is_left = (x <= surface_threshold)
        is_right = (x >= w - surface_threshold)
        is_bottom = (y <= surface_threshold)
        is_top = (y >= h - surface_threshold)

        # 1) Corners
        if is_left && is_bottom
            # bottom-left corner: diagonal out is (-1, -1), normalized
            return [-sqrt(0.5), -sqrt(0.5)]
        elseif is_left && is_top
            # top-left corner
            return [-sqrt(0.5), sqrt(0.5)]
        elseif is_right && is_bottom
            # bottom-right corner
            return [sqrt(0.5), -sqrt(0.5)]
        elseif is_right && is_top
            # top-right corner
            return [sqrt(0.5), sqrt(0.5)]
        end

        # 2) Single edges
        if is_left
            return [-1.0, 0.0]
        elseif is_right
            return [1.0, 0.0]
        elseif is_bottom
            return [0.0, -1.0]
        elseif is_top
            return [0.0, 1.0]
        end

        # 3) Interior
        return [0.0, 0.0]
    end

    computed_normals = copy(system.cache.surface_normal)

    # Normalize computed normals for any particle where it's nonzero
    for i in 1:nparticles
        nc = norm(computed_normals[:, i])
        if nc > eps()
            computed_normals[:, i] /= nc
        end
    end

    # Compare computed normals vs. expected normals
    for i in 1:nparticles
        pos = coordinates[:, i]
        exp_normal = expected_rect_normal(pos, width, height, surface_threshold)
        nexp = norm(exp_normal)

        # ignore interior values since the normals are just approximation and will have nonzero values in the interior
        if nexp > 0.1
            # Expected = nonzero => direction check
            dot_val = dot(exp_normal, -computed_normals[:, i])
            # They should be close to parallel and same direction => dot ~ 1.0
            @test isapprox(dot_val, 1.0; atol=0.1)
        end
    end

    function is_corner(x, y; tol=0.5 * particle_spacing)
        isleft = (x <= tol)
        isright = (x >= width - tol)
        isbottom = (y <= tol)
        istop = (y >= height - tol)
        return (isleft || isright) && (isbottom || istop)
    end

    curvature = system.cache.curvature

    for i in 1:nparticles
        x, y = coordinates[:, i]

        # Skip corners, which are theoretically infinite curvature
        if is_corner(x, y)
            continue
        end

        # Just test the interior for now since the normal values are unreliable
        if norm(computed_normals[:, i]) < 0.5
            @test isapprox(curvature[i], 0.0; atol=1e-2)
        end
    end
end
