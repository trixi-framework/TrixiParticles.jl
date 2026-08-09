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
                             surface_normal_method=ColorfieldSurfaceNormal(),
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
                                         surface_normal_method,
                                         reference_particle_spacing=particle_spacing,
                                         surface_tension)

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

function compute_and_test_surface_values(system, semi, ode; NDIMS=2)
    v0_ode, u0_ode = ode.u0.x
    v = TrixiParticles.wrap_v(v0_ode, system, semi)
    u = TrixiParticles.wrap_u(u0_ode, system, semi)

    # Compute the surface normals
    TrixiParticles.compute_surface_normal!(system, system.surface_normal_method, v, u,
                                           v0_ode, u0_ode, semi, 0.0)

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

@testset "Corrected C-CSF interface geometry" begin
    particle_spacing = 0.05
    radius = 0.5
    reference_density = 1000.0
    smoothing_kernel = WendlandC2Kernel{2}()
    smoothing_length = 1.4particle_spacing
    fluid = SphereShape(particle_spacing, radius, (0.0, 0.0), reference_density;
                        sphere_type=RoundSphere())
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density,
                                       exponent=7)
    surface_tension = SurfaceTensionMorris(; surface_tension_coefficient=1.0)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation, surface_tension,
                                         surface_normal_method=CorrectedCSFSurfaceNormal(),
                                         reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(system)
    ode = semidiscretize(semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    cache = system.cache
    active = findall(>(0), cache.interface_activity)
    @test !isempty(active)
    @test all(isfinite, cache.ccsf_minimum_eigenvalue)
    @test 0.4 < minimum(cache.ccsf_minimum_eigenvalue) < 0.6
    @test maximum(cache.ccsf_minimum_eigenvalue) > 0.99
    @test all(isfinite, cache.surface_normal)
    @test all(isfinite, cache.curvature)
    @test all(>=(0), cache.delta_s)
    @test all(active) do particle
        dot(TrixiParticles.surface_normal(system, particle),
            fluid.coordinates[:, particle]) > 0
    end

    weighted_curvature = sum(cache.curvature[active] .* cache.delta_s[active]) /
                         sum(cache.delta_s[active])
    @test isapprox(weighted_curvature, inv(radius); rtol=0.15)

    system_data = Dict{String, Any}()
    TrixiParticles.add_system_data!(system_data, system.surface_normal_method)
    @test system_data["surface_normal_method"]["model"] ==
          "CorrectedCSFSurfaceNormal"

    @test_throws ArgumentError WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                           smoothing_length,
                                                           density_calculator=ContinuityDensity(),
                                                           state_equation,
                                                           surface_tension=SurfaceTensionMomentumMorris(),
                                                           surface_normal_method=CorrectedCSFSurfaceNormal(),
                                                           reference_particle_spacing=particle_spacing)
end

@testset "Corrected C-CSF 3D curvature" begin
    particle_spacing = 0.05
    radius = 0.5
    reference_density = 1000.0
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    fluid = SphereShape(particle_spacing, radius, (0.0, 0.0, 0.0), reference_density;
                        sphere_type=RoundSphere())
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation=StateEquationCole(;
                                                                          sound_speed=10.0,
                                                                          reference_density,
                                                                          exponent=7),
                                         surface_tension=SurfaceTensionMorris(;
                                                                              surface_tension_coefficient=1.0),
                                         surface_normal_method=CorrectedCSFSurfaceNormal(),
                                         reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(system)
    ode = semidiscretize(semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    cache = system.cache
    active = findall(>(0), cache.interface_activity)
    @test !isempty(active)
    @test all(active) do particle
        dot(TrixiParticles.surface_normal(system, particle),
            fluid.coordinates[:, particle]) > 0
    end
    @test minimum(cache.curvature[active]) > 0

    weighted_curvature = sum(cache.curvature[active] .* cache.delta_s[active]) /
                         sum(cache.delta_s[active])
    @test isapprox(weighted_curvature, 2 / radius; rtol=0.15)
end

@testset "Shepard-smoothed CSS normals" begin
    particle_spacing = 0.1
    reference_density = 1000.0
    fluid = SphereShape(particle_spacing, 0.5, (0.0, 0.0, 0.0), reference_density;
                        sphere_type=RoundSphere())
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density,
                                       exponent=7)
    surface_tension = SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
    normal_method = ColorfieldSurfaceNormal(; ideal_density_threshold=0.95,
                                            normal_smoothing=true)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation, surface_tension,
                                         surface_normal_method=normal_method,
                                         reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    active = findall(>(0), system.cache.interface_activity)
    @test !isempty(active)
    @test haskey(system.cache, :smoothed_surface_normal)
    @test haskey(system.cache, :normal_smoothing_weight)
    @test all(isfinite, system.cache.smoothed_surface_normal)
    @test all(isfinite, system.cache.normal_smoothing_weight)
    @test all(active) do particle
        isapprox(norm(TrixiParticles.surface_tension_normal(system, particle)), 1;
                 atol=1.0e-12)
    end

    raw_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                             density_calculator=ContinuityDensity(),
                                             state_equation, surface_tension,
                                             surface_normal_method=ColorfieldSurfaceNormal(;
                                                                                           ideal_density_threshold=0.95),
                                             reference_particle_spacing=particle_spacing)
    raw_semi = Semidiscretization(raw_system)
    raw_ode = semidiscretize(raw_semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(raw_ode.u0.x..., raw_semi, 0.0)

    # Smoothing changes only the capillary direction, not raw geometry or activity.
    @test !haskey(raw_system.cache, :smoothed_surface_normal)
    @test !haskey(raw_system.cache, :normal_smoothing_weight)
    @test system.cache.surface_normal ≈ raw_system.cache.surface_normal
    @test system.cache.interface_activity ≈ raw_system.cache.interface_activity
    @test system.cache.delta_s ≈ raw_system.cache.delta_s
    differences = [norm(TrixiParticles.surface_tension_normal(system, particle) -
                        TrixiParticles.surface_normal(system, particle))
                   for particle in active]
    candidate = active[argmax(differences)]
    @test maximum(differences) > 1.0e-4

    inactive = setdiff(eachparticle(system), active)
    @test all(particle -> iszero(TrixiParticles.surface_tension_normal(system, particle)),
              inactive)

    grad_kernel = SVector(0.3, -0.4, 0.2)
    normal = TrixiParticles.surface_tension_normal(system, candidate)
    delta_s = system.cache.delta_s[candidate]
    expected_stress_gradient = delta_s *
                               (grad_kernel - normal * dot(normal, grad_kernel))
    @test TrixiParticles.surface_stress_times_gradient(system, candidate, grad_kernel) ≈
          expected_stress_gradient

    vtk = Dict{String, Any}()
    GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        TrixiParticles.write2vtk!(vtk, v, u, 0.0, system)
    end
    expected_stress = delta_s *
                      (Matrix{Float64}(I, 3, 3) - normal * transpose(normal))
    @test vtk["surf_normal"][candidate] ≈ TrixiParticles.surface_normal(system, candidate)
    @test vtk["surface_tension_normal"][candidate] ≈ normal
    @test vtk["surface_stress_tensor"][:, :, candidate] ≈ expected_stress

    system.cache.surface_normal .= reshape([1.0, 0.0, 0.0], 3, 1)
    system.cache.interface_activity .= 1
    GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        TrixiParticles.smooth_surface_normals!(system, normal_method, v, u, semi)
    end
    @test all(particle -> TrixiParticles.surface_tension_normal(system, particle) ==
                          SVector(1.0, 0.0, 0.0), eachparticle(system))

    system.cache.surface_normal .= 0
    GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        TrixiParticles.smooth_surface_normals!(system, normal_method, v, u, semi)
    end
    @test all(iszero, system.cache.smoothed_surface_normal)
    @test all(isfinite, system.cache.smoothed_surface_normal)
end

@testset "CSS flat-pool geometry" begin
    particle_spacing = 0.1
    reference_density = 1000.0
    smoothing_kernel = WendlandC2Kernel{2}()
    smoothing_length = 1.4particle_spacing
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density,
                                       exponent=1)
    fluid = RectangularShape(particle_spacing, (9, 6), (0.0, 0.0);
                             density=reference_density)
    normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1,
                                            interface_threshold=0.01,
                                            ideal_density_threshold=0.9)
    surface_tension = SurfaceTensionMomentumMorris(; surface_tension_coefficient=0.072)
    fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=ContinuityDensity(),
                                               state_equation, surface_tension,
                                               surface_normal_method=normal_method,
                                               reference_particle_spacing=particle_spacing)

    # The top wall row continues the fluid lattice one spacing below the bottom fluid row.
    wall = RectangularShape(particle_spacing, (9, 3), (0.0, -0.3);
                            density=reference_density)
    boundary_model = BoundaryModelDummyParticles(wall; fluid_system,
                                                 boundary_density_calculator=AdamiPressureExtrapolation())
    boundary_system = WallBoundarySystem(wall, boundary_model)
    semi = Semidiscretization(fluid_system, boundary_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    coordinates = fluid.coordinates
    particle_at(position) = findfirst(particle -> coordinates[:, particle] == position,
                                      axes(coordinates, 2))
    bottom_center = particle_at([0.45, 0.05])
    interior_center = particle_at([0.45, 0.25])
    top_center = particle_at([0.45, 0.55])
    centerline_particles = [bottom_center, interior_center, top_center]

    acceleration = GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        v_boundary = TrixiParticles.wrap_v(v_ode, boundary_system, semi)
        u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
        dv = zeros(eltype(v), size(v))
        TrixiParticles.interact!(dv, v, u, v, u, fluid_system, fluid_system, semi)
        TrixiParticles.interact!(dv, v, u, v_boundary, u_boundary, fluid_system,
                                 boundary_system, semi)
        Array(dv[1:2, :])
    end

    # Wall particles complete the support moment without carrying capillary stress.
    @test fluid_system.cache.divergence_correction[bottom_center] >= 0.9
    @test fluid_system.cache.interface_activity[bottom_center] == 0
    @test fluid_system.cache.delta_s[bottom_center] == 0
    @test fluid_system.cache.delta_s[top_center] > 0
    @test iszero(fluid_system.cache.delta_s[interior_center])
    @test maximum(abs, acceleration[:, centerline_particles]) < 1.0e-12
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
                                   surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                                 ideal_density_threshold=0.9),
                                   wall=true, walldistance=particle_spacing,
                                   boundary_system_type=:wall)

    rigid_system, rigid_boundary, rigid_semi,
    rigid_ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                                    SurfaceTensionMorris(surface_tension_coefficient=0.072);
                                    NDIMS, smoothing_length, smoothing_kernel,
                                    surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                                  ideal_density_threshold=0.9),
                                    wall=true, walldistance=particle_spacing,
                                    boundary_system_type=:rigid)

    free_system, _, free_semi,
    free_ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                                   SurfaceTensionMorris(surface_tension_coefficient=0.072);
                                   NDIMS, smoothing_length, smoothing_kernel,
                                   surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                                 ideal_density_threshold=0.9))

    compute_and_test_surface_values(wall_system, wall_semi, wall_ode; NDIMS)
    compute_and_test_surface_values(rigid_system, rigid_semi, rigid_ode; NDIMS)
    compute_and_test_surface_values(free_system, free_semi, free_ode; NDIMS)

    @test isapprox(rigid_boundary.boundary_model.cache.initial_colorfield,
                   wall_boundary.boundary_model.cache.initial_colorfield,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
    @test isapprox(rigid_system.cache.surface_normal,
                   wall_system.cache.surface_normal,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
    @test isapprox(rigid_system.cache.neighbor_count,
                   wall_system.cache.neighbor_count,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
    @test all(isfinite, wall_system.cache.support_moment)
    @test maximum(abs, wall_system.cache.support_moment) > 0
    @test isapprox(rigid_system.cache.support_moment,
                   wall_system.cache.support_moment,
                   rtol=sqrt(eps()), atol=sqrt(eps()))
    @test maximum(abs,
                  wall_system.cache.support_moment - free_system.cache.support_moment) >
          sqrt(eps())
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
                                      surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
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
                                      surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
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
