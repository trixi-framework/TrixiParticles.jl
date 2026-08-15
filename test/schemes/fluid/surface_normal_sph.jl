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
                                                 AdamiPressureExtrapolation(),
                                                 kernel,
                                                 smoothing_length;
                                                 state_equation,
                                                 correction=nothing,
                                                 reference_particle_spacing=particle_spacing)

    boundary_system = WallBoundarySystem(wall, boundary_model, adhesion_coefficient=0.0)
    return boundary_system
end

function create_rigid_boundary_system(coordinates, particle_spacing, state_equation, kernel,
                                      smoothing_length, NDIMS, walldistance)
    # Reuse the same particle layout as the wall-boundary helper so the rigid/body and wall
    # variants should generate identical colorfield data.
    wall_system = create_boundary_system(coordinates, particle_spacing, state_equation,
                                         kernel, smoothing_length, NDIMS, walldistance)

    rigid_model = BoundaryModelDummyParticles(wall_system.initial_condition.density,
                                              wall_system.initial_condition.mass,
                                              AdamiPressureExtrapolation(), kernel,
                                              smoothing_length; state_equation,
                                              correction=nothing,
                                              reference_particle_spacing=particle_spacing)

    return RigidBodySystem(wall_system.initial_condition;
                           boundary_model=rigid_model,
                           acceleration=ntuple(_ -> 0.0, NDIMS))
end

function create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                             surface_tension;
                             surface_method=ColorfieldSurfaceNormal(), color_value=1,
                             NDIMS=2, smoothing_length=1.0, wall=false, walldistance=0.0,
                             boundary_system_type=:wall,
                             smoothing_kernel=SchoenbergCubicSplineKernel{NDIMS}())
    tspan = (0.0, 0.01)

    fluid = InitialCondition(; coordinates, velocity, mass, density, particle_spacing)

    state_equation = StateEquationCole(sound_speed=10.0,
                                       reference_density=1000.0,
                                       exponent=1)

    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=SummationDensity(),
                                         state_equation,
                                         surface_method,
                                         reference_particle_spacing=particle_spacing,
                                         surface_tension, color_value)

    if wall
        boundary_system = if boundary_system_type == :wall
            create_boundary_system(coordinates, particle_spacing, state_equation,
                                   smoothing_kernel, smoothing_length, NDIMS,
                                   walldistance)
        elseif boundary_system_type == :rigid
            create_rigid_boundary_system(coordinates, particle_spacing, state_equation,
                                         smoothing_kernel, smoothing_length, NDIMS,
                                         walldistance)
        else
            error("unsupported boundary_system_type: $boundary_system_type")
        end
        semi = Semidiscretization(system, boundary_system)
    else
        semi = Semidiscretization(system)
        boundary_system = nothing
    end

    ode = semidiscretize(semi, tspan)
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    return system, boundary_system, semi, ode
end

@testset "Multicolor fluid interfaces" begin
    particle_spacing = 0.1
    smoothing_length = 0.15
    y_coordinates = collect(-0.5:particle_spacing:0.5)
    coordinates_a = hcat(([x, y] for x in -0.5:particle_spacing:-0.1
                          for y in y_coordinates)...)
    coordinates_b = hcat(([x, y] for x in 0.0:particle_spacing:0.5
                          for y in y_coordinates)...)
    smoothing_kernel = WendlandC2Kernel{2}()
    state_equation = StateEquationCole(sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    normal_method = ColorfieldSurfaceNormal(interface_threshold=1.0e-6)

    function interface_normal(color_a, color_b)
        initial_condition_a = InitialCondition(; coordinates=coordinates_a,
                                               density=fill(1000.0,
                                                            size(coordinates_a, 2)),
                                               particle_spacing)
        initial_condition_b = InitialCondition(; coordinates=coordinates_b,
                                               density=fill(1000.0,
                                                            size(coordinates_b, 2)),
                                               particle_spacing)

        system_a = WeaklyCompressibleSPHSystem(initial_condition_a; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=SummationDensity(),
                                               state_equation,
                                               surface_normal_method=normal_method,
                                               reference_particle_spacing=particle_spacing,
                                               color_value=color_a)
        # A fluid contributes its color even when it does not compute its own normals.
        system_b = WeaklyCompressibleSPHSystem(initial_condition_b; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=SummationDensity(),
                                               state_equation, color_value=color_b)
        semi = Semidiscretization(system_a, system_b)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

        v_a = TrixiParticles.wrap_v(v_ode, system_a, semi)
        u_a = TrixiParticles.wrap_u(u_ode, system_a, semi)
        TrixiParticles.compute_surface_normal!(system_a, normal_method, v_a, u_a,
                                               v_ode, u_ode, semi, 0.0)

        interface_particle = argmin(eachindex(eachcol(coordinates_a))) do particle
            abs(coordinates_a[1, particle] + particle_spacing) +
            abs(coordinates_a[2, particle])
        end

        return TrixiParticles.surface_normal(system_a, interface_particle), system_b
    end

    increasing_normal, non_normal_neighbor = interface_normal(0, 2)
    unit_jump_normal, _ = interface_normal(0, 1)
    decreasing_normal, _ = interface_normal(2, 0)
    equal_color_normal, _ = interface_normal(1, 1)
    canceling_labels_normal, _ = interface_normal(-1, 1)

    @test isnothing(non_normal_neighbor.surface_normal_method)
    @test increasing_normal[1] > 0
    @test decreasing_normal[1] < 0
    @test abs(increasing_normal[2]) < 100eps()
    @test abs(decreasing_normal[2]) < 100eps()
    @test isapprox(norm(increasing_normal), norm(decreasing_normal); rtol=1.0e-12)
    @test isapprox(norm(increasing_normal), 2 * norm(unit_jump_normal); rtol=1.0e-12)
    @test iszero(equal_color_normal)
    @test canceling_labels_normal[1] > 0
end

@testset "Standalone surface-normal thresholds" begin
    particle_spacing = 0.1
    coordinates = hcat(([x, y] for x in 0.0:particle_spacing:0.6
                        for y in 0.0:particle_spacing:0.6)...)
    velocity = zeros(2, size(coordinates, 2))
    density = fill(1000.0, size(coordinates, 2))
    mass = fill(10.0, size(coordinates, 2))

    system, boundary, semi,
    ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                              nothing;
                              smoothing_length=0.15,
                              surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=1.0e6))
    @test isnothing(boundary)
    @test all(iszero, system.cache.surface_normal)

    system, boundary, semi,
    ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                              nothing;
                              smoothing_length=0.15,
                              surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.0))
    @test any(!iszero, system.cache.surface_normal)

    system, boundary, semi,
    ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                              nothing;
                              smoothing_length=0.15,
                              surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.0,
                                                                            ideal_density_threshold=0.01))
    @test isnothing(boundary)
    @test all(iszero, system.cache.surface_normal)

    @test_throws ArgumentError create_fluid_system(coordinates, velocity, mass, density,
                                                   particle_spacing, nothing;
                                                   smoothing_length=0.15, color_value=0,
                                                   wall=true, walldistance=particle_spacing)
end

function compute_and_test_surface_values(system, semi, ode; NDIMS=2)
    v0_ode, u0_ode = ode.u0.x
    v = TrixiParticles.wrap_v(v0_ode, system, semi)
    u = TrixiParticles.wrap_u(u0_ode, system, semi)

    TrixiParticles.compute_surface!(system, system.surface_method, v, u,
                                    v0_ode, u0_ode, semi, 0.0)

    TrixiParticles.remove_invalid_normals!(system, system.surface_tension,
                                           system.surface_method)

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

@testset verbose=true "Colorfield Surface Detection" begin
    normal_method = ColorfieldSurfaceNormal(ideal_density_threshold=0.9)
    detection_method = ColorfieldSurfaceDetection(ideal_density_threshold=0.9)
    @test ColorfieldSurfaceNormal() == ColorfieldSurfaceNormal(0.1, 0.01, 0.0)
    @test normal_method.interpolation_surface_threshold == 0.45
    @test detection_method.interpolation_surface_threshold == 0.45
    @test TrixiParticles.computes_surface_normal(normal_method)
    @test !TrixiParticles.computes_surface_normal(detection_method)

    @test_throws ArgumentError ColorfieldSurfaceNormal(boundary_contact_threshold=-0.1)
    @test_throws ArgumentError ColorfieldSurfaceNormal(interface_threshold=Inf)
    @test_throws ArgumentError ColorfieldSurfaceNormal(interface_taper_start=1.0)
    @test_throws ArgumentError ColorfieldSurfaceNormal(interpolation_surface_threshold=1.1)
    @test_throws ArgumentError ColorfieldSurfaceDetection(interface_threshold=-0.1)
    @test_throws ArgumentError ColorfieldSurfaceDetection(interface_threshold="invalid")

    particle_spacing = 0.1
    coordinates = RectangularShape(particle_spacing, (21, 11), (0.0, 0.0),
                                   density=1000.0)
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    system, _, semi,
    ode = create_fluid_system(coordinates.coordinates, coordinates.velocity,
                              coordinates.mass, coordinates.density, particle_spacing,
                              nothing; smoothing_length=1.5 * particle_spacing,
                              smoothing_kernel, surface_method=normal_method)

    activity = system.cache.surface_activity
    x = coordinates.coordinates
    min_x, max_x = extrema(view(x, 1, :))
    min_y, max_y = extrema(view(x, 2, :))
    interior = [particle
                for particle in eachindex(activity)
                if min_x + 3particle_spacing < x[1, particle] <
                   max_x - 3particle_spacing &&
                   min_y + 3particle_spacing < x[2, particle] <
                   max_y - 3particle_spacing]
    top_surface = [particle
                   for particle in eachindex(activity)
                   if x[2, particle] == max_y &&
                      min_x + 3particle_spacing < x[1, particle] <
                      max_x - 3particle_spacing]

    @test all(isfinite, activity)
    @test all(iszero, activity[interior])
    @test all(==(1), activity[top_surface])

    detection_system = WeaklyCompressibleSPHSystem(coordinates;
                                                   smoothing_kernel,
                                                   smoothing_length=1.5 * particle_spacing,
                                                   density_calculator=SummationDensity(),
                                                   state_equation=system.state_equation,
                                                   surface_method=detection_method,
                                                   reference_particle_spacing=particle_spacing)
    detection_semi = Semidiscretization(detection_system)
    detection_ode = semidiscretize(detection_semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(detection_ode.u0.x..., detection_semi, 0.0)
    @test !haskey(detection_system.cache, :surface_normal)
    @test detection_system.cache.surface_activity == activity
    @test isapprox(detection_system.cache.surface_gradient, system.cache.surface_normal;
                   rtol=10eps(), atol=10eps())

    edac_system = EntropicallyDampedSPHSystem(coordinates;
                                              smoothing_kernel,
                                              smoothing_length=1.5 * particle_spacing,
                                              sound_speed=10.0,
                                              density_calculator=SummationDensity(),
                                              surface_method=detection_method,
                                              reference_particle_spacing=particle_spacing)
    edac_semi = Semidiscretization(edac_system)
    edac_ode = semidiscretize(edac_semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(edac_ode.u0.x..., edac_semi, 0.0)
    @test edac_system.cache.surface_activity == activity

    iisph_system = ImplicitIncompressibleSPHSystem(coordinates;
                                                   smoothing_kernel,
                                                   smoothing_length=1.5 * particle_spacing,
                                                   reference_density=1000.0,
                                                   time_step=0.001,
                                                   surface_method=detection_method,
                                                   reference_particle_spacing=particle_spacing)
    iisph_semi = Semidiscretization(iisph_system)
    iisph_ode = semidiscretize(iisph_semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(iisph_ode.u0.x..., iisph_semi, 0.0)
    @test iisph_system.cache.surface_activity == activity

    @test_throws ArgumentError WeaklyCompressibleSPHSystem(coordinates;
                                                           smoothing_kernel,
                                                           smoothing_length=1.5 *
                                                                            particle_spacing,
                                                           density_calculator=SummationDensity(),
                                                           state_equation=system.state_equation,
                                                           surface_tension=SurfaceTensionMorris(),
                                                           surface_method=detection_method,
                                                           reference_particle_spacing=particle_spacing)
    cohesion_system = WeaklyCompressibleSPHSystem(coordinates;
                                                  smoothing_kernel,
                                                  smoothing_length=1.5 * particle_spacing,
                                                  density_calculator=SummationDensity(),
                                                  state_equation=system.state_equation,
                                                  surface_tension=CohesionForceAkinci())
    @test isnothing(cohesion_system.surface_method)
    @test !haskey(cohesion_system.cache, :surface_activity)

    corrected_system = WeaklyCompressibleSPHSystem(coordinates;
                                                   smoothing_kernel,
                                                   smoothing_length=1.5 * particle_spacing,
                                                   density_calculator=SummationDensity(),
                                                   state_equation=system.state_equation,
                                                   correction=GradientCorrection(),
                                                   surface_method=normal_method,
                                                   reference_particle_spacing=particle_spacing)
    corrected_semi = Semidiscretization(corrected_system)
    corrected_ode = semidiscretize(corrected_semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(corrected_ode.u0.x..., corrected_semi, 0.0)
    @test corrected_system.cache.surface_activity == activity
    @test isapprox(corrected_system.cache.surface_normal, system.cache.surface_normal;
                   rtol=10eps(), atol=10eps())

    v_ode, u_ode = ode.u0.x
    @test TrixiParticles.surface_activity(system, nothing, nothing, v_ode, u_ode,
                                          semi, 0.0) == activity
    @test TrixiParticles.surface_normal(detection_system, nothing, nothing,
                                        detection_ode.u0.x..., detection_semi, 0.0) ===
          nothing

    metadata = Dict{String, Any}()
    TrixiParticles.add_system_data!(metadata, normal_method)
    @test metadata["surface_method"]["computes_surface_normal"]
    @test metadata["surface_method"]["interpolation_surface_threshold"] == 0.45

    mktempdir() do output_directory
        trixi2vtk(ode.u0, semi, 0.0; output_directory,
                  prefix="surface_detection", overwrite=true)
        vtk_data = vtk2trixi(joinpath(output_directory,
                                      "surface_detection_fluid_1_current.vtu"))
        @test vtk_data.surface_activity == activity
        @test hasproperty(vtk_data, :surf_normal)

        trixi2vtk(detection_ode.u0, detection_semi, 0.0; output_directory,
                  prefix="detection_only", overwrite=true)
        detection_vtk_data = vtk2trixi(joinpath(output_directory,
                                                "detection_only_fluid_1_current.vtu"))
        @test detection_vtk_data.surface_activity == activity
        @test !hasproperty(detection_vtk_data, :surf_normal)
    end
end

@testset verbose=true "Multicolor Surface Activity" begin
    particle_spacing = 0.1
    smoothing_length = 0.15
    y_coordinates = collect(-0.5:particle_spacing:0.5)
    coordinates_a = hcat(([x, y] for x in -0.5:particle_spacing:-0.1
                          for y in y_coordinates)...)
    coordinates_b = hcat(([x, y] for x in 0.0:particle_spacing:0.5
                          for y in y_coordinates)...)
    smoothing_kernel = WendlandC2Kernel{2}()
    state_equation = StateEquationCole(sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)

    function interface_geometry(color_a, color_b, surface_method_)
        initial_condition_a = InitialCondition(; coordinates=coordinates_a,
                                               density=fill(1000.0,
                                                            size(coordinates_a, 2)),
                                               particle_spacing)
        initial_condition_b = InitialCondition(; coordinates=coordinates_b,
                                               density=fill(1000.0,
                                                            size(coordinates_b, 2)),
                                               particle_spacing)
        system_a = WeaklyCompressibleSPHSystem(initial_condition_a; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=SummationDensity(),
                                               state_equation,
                                               surface_method=surface_method_,
                                               reference_particle_spacing=particle_spacing,
                                               color_value=color_a)
        system_b = WeaklyCompressibleSPHSystem(initial_condition_b; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=SummationDensity(),
                                               state_equation, color_value=color_b)
        semi = Semidiscretization(system_a, system_b)
        ode = semidiscretize(semi, (0.0, 0.01))
        TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

        interface_particle = argmin(eachindex(eachcol(coordinates_a))) do particle
            abs(coordinates_a[1, particle] + particle_spacing) +
            abs(coordinates_a[2, particle])
        end
        gradient = if surface_method_ isa ColorfieldSurfaceNormal
            TrixiParticles.surface_normal(system_a, interface_particle)
        else
            TrixiParticles.extract_svector(system_a.cache.surface_gradient, system_a,
                                           interface_particle)
        end
        return gradient, TrixiParticles.surface_activity(system_a, interface_particle),
               system_b
    end

    detection_method = ColorfieldSurfaceDetection(interface_threshold=1.0e-6)
    normal_method = ColorfieldSurfaceNormal(interface_threshold=1.0e-6)
    increasing_gradient, increasing_activity,
    non_surface_neighbor = interface_geometry(0, 2, detection_method)
    unit_gradient, _, _ = interface_geometry(0, 1, detection_method)
    decreasing_gradient, decreasing_activity, _ = interface_geometry(2, 0,
                                                                     detection_method)
    equal_gradient, equal_activity, _ = interface_geometry(1, 1, detection_method)
    normal_gradient, normal_activity, _ = interface_geometry(0, 2, normal_method)

    @test isnothing(non_surface_neighbor.surface_method)
    @test increasing_gradient[1] > 0
    @test decreasing_gradient[1] < 0
    @test isapprox(norm(increasing_gradient), 2norm(unit_gradient); rtol=1.0e-12)
    @test norm(equal_gradient) < 100eps()
    @test increasing_activity == 1
    @test decreasing_activity == 1
    @test equal_activity == 0
    @test normal_activity == increasing_activity
    @test normal_gradient == increasing_gradient
end

@testset verbose=true "Rigid Dummy Boundary Matches Wall Boundary" begin
    NDIMS = 2
    particle_spacing = 0.2
    smoothing_length = 3.0 * particle_spacing
    smoothing_kernel = SchoenbergCubicSplineKernel{NDIMS}()
    radius = 1.0
    center = (0.0, 0.0)

    sphere_ic = SphereShape(particle_spacing, radius, center, 1000.0)
    coordinates = sphere_ic.coordinates
    velocity = zeros(NDIMS, size(coordinates, 2))
    mass = sphere_ic.mass
    density = sphere_ic.density

    wall_system, wall_boundary, wall_semi,
    wall_ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                                   SurfaceTensionMorris(surface_tension_coefficient=0.072);
                                   NDIMS, smoothing_length, smoothing_kernel,
                                   surface_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                          ideal_density_threshold=0.9),
                                   wall=true, walldistance=2.0, boundary_system_type=:wall)

    rigid_system, rigid_boundary, rigid_semi,
    rigid_ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                                    SurfaceTensionMorris(surface_tension_coefficient=0.072);
                                    NDIMS, smoothing_length, smoothing_kernel,
                                    surface_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                           ideal_density_threshold=0.9),
                                    wall=true, walldistance=2.0,
                                    boundary_system_type=:rigid)

    compute_and_test_surface_values(wall_system, wall_semi, wall_ode; NDIMS)
    compute_and_test_surface_values(rigid_system, rigid_semi, rigid_ode; NDIMS)

    @test isapprox(rigid_boundary.boundary_model.cache.initial_colorfield,
                   wall_boundary.boundary_model.cache.initial_colorfield,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
    @test isapprox(rigid_system.cache.surface_normal,
                   wall_system.cache.surface_normal,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
    @test isapprox(rigid_system.cache.neighbor_count,
                   wall_system.cache.neighbor_count,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
    @test isapprox(rigid_system.cache.surface_activity,
                   wall_system.cache.surface_activity,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
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
            ode = create_fluid_system(coordinates, velocity, mass, density,
                                      particle_spacing,
                                      SurfaceTensionMorris(surface_tension_coefficient=0.072);
                                      NDIMS, smoothing_length, smoothing_kernel,
                                      surface_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                             ideal_density_threshold=0.9),
                                      wall=true, walldistance=2.0)

            compute_and_test_surface_values(system, semi, ode; NDIMS)

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
            ode = create_fluid_system(coordinates, velocity, mass, density,
                                      particle_spacing,
                                      SurfaceTensionAkinci(surface_tension_coefficient=0.072);
                                      NDIMS, smoothing_length, smoothing_kernel,
                                      surface_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                             ideal_density_threshold=0.9),
                                      wall=true, walldistance=2.0)

            compute_and_test_surface_values(system, semi, ode; NDIMS)

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
    ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                              SurfaceTensionMorris(surface_tension_coefficient=0.072);
                              NDIMS, smoothing_length=1.5 * particle_spacing, wall=false,
                              walldistance=0.0)

    # Compute surface normals
    compute_and_test_surface_values(system, semi, ode; NDIMS)

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
