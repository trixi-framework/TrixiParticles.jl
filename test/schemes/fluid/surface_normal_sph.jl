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

@testset "Akinci planar-normal magnitude" begin
    particle_spacing = 1.0
    smoothing_kernel = SchoenbergCubicSplineKernel{3}()
    smoothing_length = particle_spacing
    fluid = RectangularShape(particle_spacing, (5, 5, 5), (0.0, 0.0, 0.0);
                             density=1.0)
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1.0,
                                       exponent=1)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation,
                                         surface_tension=SurfaceTensionAkinci(),
                                         reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    coordinates = fluid.coordinates
    center = findfirst(particle -> coordinates[:, particle] == [2.5, 2.5, 2.5],
                       axes(coordinates, 2))
    left_face = findfirst(particle -> coordinates[:, particle] == [0.5, 2.5, 2.5],
                          axes(coordinates, 2))
    right_face = findfirst(particle -> coordinates[:, particle] == [4.5, 2.5, 2.5],
                           axes(coordinates, 2))

    @test norm(TrixiParticles.akinci_surface_normal(system, center)) < 2e-16
    @test isapprox(TrixiParticles.akinci_surface_normal(system, left_face),
                   SVector(1.0, 0.0, 0.0); atol=0.03)
    @test isapprox(TrixiParticles.akinci_surface_normal(system, right_face),
                   SVector(-1.0, 0.0, 0.0); atol=0.03)
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
            fluid.coordinates[:, particle]) >
        0
    end

    weighted_curvature = sum(cache.curvature[active] .* cache.delta_s[active]) /
                         sum(cache.delta_s[active])
    @test isapprox(weighted_curvature, inv(radius); rtol=0.15)

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
            fluid.coordinates[:, particle]) >
        0
    end
    @test minimum(cache.curvature[active]) > 0

    weighted_curvature = sum(cache.curvature[active] .* cache.delta_s[active]) /
                         sum(cache.delta_s[active])
    @test isapprox(weighted_curvature, 2 / radius; rtol=0.15)
end

@testset "Corrected C-CSF planar boundary geometry" begin
    nonsymmetric_moment = TrixiParticles.SMatrix{2, 2}((2.0, 0.2, 0.8, 1.0))
    symmetrized_moment = (nonsymmetric_moment + transpose(nonsymmetric_moment)) / 2
    @test TrixiParticles.ccsf_minimum_eigenvalue(nonsymmetric_moment) ≈
          minimum(eigvals(Symmetric(symmetrized_moment)))
    renormalization = inv(nonsymmetric_moment)
    normal_difference = SVector(0.3, -0.4)
    kernel_direction = SVector(-0.2, 0.7)
    corrected_divergence = dot(renormalization * normal_difference, kernel_direction)
    @test TrixiParticles.ccsf_corrected_divergence(normal_difference, renormalization,
                                                   kernel_direction) ≈ corrected_divergence
    @test abs(corrected_divergence -
              dot(normal_difference, renormalization * kernel_direction)) > 0.01
    @test TrixiParticles.ccsf_lambda_difference(0.8, 1.0) ≈ 0.2
    @test TrixiParticles.ccsf_lambda_difference(0.6, 1.0) == 1.0

    particle_spacing = 0.1
    reference_density = 1000.0
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    fluid = RectangularShape(particle_spacing, (7, 7, 7), (0.0, 0.0, 0.0);
                             density=reference_density)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation=StateEquationCole(;
                                                                          sound_speed=10.0,
                                                                          reference_density,
                                                                          exponent=7),
                                         surface_tension=SurfaceTensionMorris(;
                                                                              surface_tension_coefficient=1.0),
                                         surface_normal_method=CorrectedCSFSurfaceNormal(;
                                                                                         contact_angle=90.0),
                                         reference_particle_spacing=particle_spacing)

    boundary_raw = RectangularShape(particle_spacing, (7, 7, 3),
                                    (0.0, 0.0, -3particle_spacing);
                                    density=reference_density)
    exposed = isapprox.(boundary_raw.coordinates[3, :],
                        maximum(boundary_raw.coordinates[3, :]); atol=eps())
    normals = zeros(size(boundary_raw.coordinates))
    normals[3, exposed] .= -particle_spacing / 2
    surface_measure = zeros(nparticles(boundary_raw))
    surface_measure[exposed] .= particle_spacing^2
    boundary = InitialCondition(; coordinates=boundary_raw.coordinates,
                                velocity=boundary_raw.velocity,
                                mass=boundary_raw.mass, density=boundary_raw.density,
                                pressure=boundary_raw.pressure, particle_spacing,
                                normals)
    boundary_model = BoundaryModelDummyParticles(boundary; fluid_system=system,
                                                 surface_measure)
    boundary_system = WallBoundarySystem(boundary, boundary_model)
    semi = Semidiscretization(system, boundary_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    center = argmin(eachparticle(system)) do particle
        sum(abs2, fluid.coordinates[:, particle] - [0.35, 0.35, 0.05])
    end
    cache = system.cache
    @test cache.ccsf_boundary_distance[center] ≈ particle_spacing / 2
    @test 0.9 < cache.ccsf_minimum_eigenvalue[center] < 1.1
    @test cache.interface_activity[center] == 0
    @test iszero(TrixiParticles.surface_normal(system, center))
    @test cache.curvature[center] == 0
    @test all(isfinite, cache.ccsf_minimum_eigenvalue)
    @test all(isfinite, cache.curvature)

    contact_line = filter(eachparticle(system)) do particle
        cache.interface_activity[particle] > 0 &&
            isapprox(cache.ccsf_boundary_distance[particle], particle_spacing / 2;
                     atol=eps())
    end
    @test !isempty(contact_line)
    @test maximum(contact_line) do particle
        abs(TrixiParticles.surface_normal(system, particle)[3])
    end < 0.05
end

@testset "Corrected C-CSF hemispherical contact" begin
    particle_spacing = 0.05
    radius = 0.5
    reference_density = 1000.0
    full_sphere = SphereShape(particle_spacing, radius, (0.0, 0.0, 0.0),
                              reference_density; sphere_type=RoundSphere())
    keep = findall(>(0), full_sphere.coordinates[3, :])
    fluid = InitialCondition(; coordinates=full_sphere.coordinates[:, keep],
                             velocity=full_sphere.velocity[:, keep],
                             mass=full_sphere.mass[keep], density=full_sphere.density[keep],
                             pressure=full_sphere.pressure[keep], particle_spacing)
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation=StateEquationCole(;
                                                                          sound_speed=10.0,
                                                                          reference_density,
                                                                          exponent=7),
                                         surface_tension=SurfaceTensionMorris(;
                                                                              surface_tension_coefficient=1.0),
                                         surface_normal_method=CorrectedCSFSurfaceNormal(;
                                                                                         contact_angle=90.0),
                                         reference_particle_spacing=particle_spacing)

    boundary_raw = RectangularShape(particle_spacing, (22, 22, 3),
                                    (-0.55, -0.55, -3particle_spacing);
                                    density=reference_density)
    exposed = isapprox.(boundary_raw.coordinates[3, :],
                        maximum(boundary_raw.coordinates[3, :]); atol=eps())
    normals = zeros(size(boundary_raw.coordinates))
    normals[3, exposed] .= -particle_spacing / 2
    surface_measure = zeros(nparticles(boundary_raw))
    surface_measure[exposed] .= particle_spacing^2
    boundary = InitialCondition(; coordinates=boundary_raw.coordinates,
                                velocity=boundary_raw.velocity,
                                mass=boundary_raw.mass, density=boundary_raw.density,
                                pressure=boundary_raw.pressure, particle_spacing,
                                normals)
    boundary_model = BoundaryModelDummyParticles(boundary; fluid_system=system,
                                                 surface_measure)
    boundary_system = WallBoundarySystem(boundary, boundary_model)
    semi = Semidiscretization(system, boundary_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    cache = system.cache
    active = findall(>(0), cache.interface_activity)
    support = 2smoothing_length
    contact = filter(active) do particle
        cache.ccsf_boundary_distance[particle] < support
    end
    @test !isempty(contact)
    @test minimum(cache.curvature[contact]) > 0
    weighted_curvature = sum(cache.curvature[active] .* cache.delta_s[active]) /
                         sum(cache.delta_s[active])
    contact_curvature = sum(cache.curvature[contact] .* cache.delta_s[contact]) /
                        sum(cache.delta_s[contact])
    @test isapprox(weighted_curvature, 2 / radius; rtol=0.1)
    @test isapprox(contact_curvature, 2 / radius; rtol=0.15)
end

@testset "Shepard-smoothed CSS normals" begin
    particle_spacing = 0.1
    reference_density = 1000.0
    fluid = SphereShape(particle_spacing, 0.5, (0.0, 0.0, 0.0), reference_density;
                        sphere_type=RoundSphere())
    system = WeaklyCompressibleSPHSystem(fluid;
                                         smoothing_kernel=WendlandC2Kernel{3}(),
                                         smoothing_length=1.4particle_spacing,
                                         density_calculator=ContinuityDensity(),
                                         state_equation=StateEquationCole(;
                                                                          sound_speed=10.0,
                                                                          reference_density,
                                                                          exponent=7),
                                         surface_tension=SurfaceTensionMomentumMorris(;
                                                                                      surface_tension_coefficient=1.0),
                                         surface_normal_method=ColorfieldSurfaceNormal(;
                                                                                       ideal_density_threshold=0.95,
                                                                                       normal_smoothing=true),
                                         reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(system)
    ode = semidiscretize(semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

    active = findall(>(0), system.cache.interface_activity)
    @test !isempty(active)
    @test all(isfinite, system.cache.smoothed_surface_normal)
    @test all(isfinite, system.cache.normal_smoothing_weight)
    @test all(active) do particle
        isapprox(norm(TrixiParticles.surface_tension_normal(system, particle)), 1;
                 atol=1.0e-12)
    end
    @test all(active) do particle
        dot(TrixiParticles.surface_normal(system, particle),
            fluid.coordinates[:, particle]) < 0
    end

    raw_system = WeaklyCompressibleSPHSystem(fluid;
                                             smoothing_kernel=WendlandC2Kernel{3}(),
                                             smoothing_length=1.4particle_spacing,
                                             density_calculator=ContinuityDensity(),
                                             state_equation=StateEquationCole(;
                                                                              sound_speed=10.0,
                                                                              reference_density,
                                                                              exponent=7),
                                             surface_tension=SurfaceTensionMomentumMorris(;
                                                                                          surface_tension_coefficient=1.0),
                                             surface_normal_method=ColorfieldSurfaceNormal(;
                                                                                           ideal_density_threshold=0.95),
                                             reference_particle_spacing=particle_spacing)
    raw_semi = Semidiscretization(raw_system)
    raw_ode = semidiscretize(raw_semi, (0.0, 0.01))
    TrixiParticles.update_systems_and_nhs(raw_ode.u0.x..., raw_semi, 0.0)

    # Smoothing is a capillary-model choice and must not alter the raw normal used by PST.
    @test system.cache.surface_normal ≈ raw_system.cache.surface_normal
    @test maximum(active) do particle
        norm(TrixiParticles.surface_tension_normal(system, particle) -
             TrixiParticles.surface_normal(system, particle))
    end > 1.0e-4
end

# With an explicit finite contact threshold, the colorfield-gradient normal extends the
# fluid-only formulation of Akinci et al. (2013) by including boundary neighbors. Fluid
# particles resting on a wetted, lattice-continuing wall are thereby treated like interior
# particles with near-zero normals, while a distant wall leaves the free-surface normal
# untouched. This test also pins the fluid-only Akinci default.
@testset "Akinci wall-contact normals" begin
    function build_fluid_over_wall(wall_offset;
                                   surface_normal_method=ColorfieldSurfaceNormal())
        particle_spacing = 0.2
        # Compact support of 1.6 particle spacings, so that only the first missing fluid
        # row below the bottom row can be replaced by wall contributions
        smoothing_length = 0.8 * particle_spacing
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=1000.0,
                                           exponent=1)

        fluid = RectangularShape(particle_spacing, (8, 6), (0.0, 0.0); density=1000.0)
        fluid_sys = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                                density_calculator=SummationDensity(),
                                                state_equation,
                                                surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.072),
                                                surface_normal_method,
                                                reference_particle_spacing=particle_spacing)

        # Wall on the same lattice as the fluid, with the top particle row `wall_offset`
        # below the bottom fluid row. `wall_offset == particle_spacing` continues the
        # fluid lattice, as the boundaries of `RectangularTank` do.
        wall_thickness = 4 * particle_spacing
        ymin = minimum(fluid.coordinates[2, :])
        wall = RectangularShape(particle_spacing, (8, 4),
                                (0.0,
                                 ymin - wall_offset - wall_thickness +
                                 particle_spacing / 2); density=1000.0)
        boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length;
                                                     state_equation,
                                                     reference_particle_spacing=particle_spacing)
        boundary_sys = WallBoundarySystem(wall, boundary_model, adhesion_coefficient=0.0)

        semi_ = Semidiscretization(fluid_sys, boundary_sys)
        ode_ = semidiscretize(semi_, (0.0, 0.01))
        v_ode_, u_ode_ = ode_.u0.x
        TrixiParticles.update_systems_and_nhs(v_ode_, u_ode_, semi_, 0.0)

        find(coords) = findfirst(p -> isapprox(fluid.coordinates[:, p], coords;
                                               atol=1e-10),
                                 axes(fluid.coordinates, 2))
        bottom_center = find([0.7, 0.1])
        top_center = find([0.7, 1.1])

        return (TrixiParticles.akinci_surface_normal(fluid_sys, bottom_center),
                TrixiParticles.akinci_surface_normal(fluid_sys, top_center))
    end

    # Wetted wall continuing the fluid lattice and wall far outside the compact support
    n_bottom_wetted, n_top_wetted = build_fluid_over_wall(0.2)
    n_bottom_far, n_top_far = build_fluid_over_wall(2.0)

    # With a distant wall, the bottom row is a free surface with an inward (upward) normal
    @test n_bottom_far[2] > 0.99 * norm(n_bottom_far)
    @test norm(n_bottom_far) > 0.5

    # The wetted wall replaces the missing fluid neighbors, so the bottom row is treated
    # like the fluid interior
    @test norm(n_bottom_wetted) < 0.05 * norm(n_bottom_far)

    # The free surface at the top is unaffected by the wall in both cases
    @test isapprox(n_top_wetted, n_top_far; rtol=sqrt(eps()))
    @test n_top_far[2] < -0.99 * norm(n_top_far)

    n_bottom_default,
    n_top_default = build_fluid_over_wall(0.2;
                                          surface_normal_method=nothing)
    n_bottom_fluid_only,
    n_top_fluid_only = build_fluid_over_wall(0.2;
                                             surface_normal_method=ColorfieldSurfaceNormal(boundary_contact_threshold=Inf))
    @test n_bottom_default == n_bottom_fluid_only
    @test n_top_default == n_top_fluid_only
end

@testset "CSS flat-pool geometry" begin
    function build_flat_pool(contact_model)
        particle_spacing = 0.1
        reference_density = 1000.0
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.4 * particle_spacing
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density,
                                           exponent=1)
        fluid = RectangularShape(particle_spacing, (9, 6), (0.0, 0.0);
                                 density=reference_density)
        normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1,
                                                interface_threshold=0.01,
                                                ideal_density_threshold=0.9,
                                                contact_model)
        surface_tension = SurfaceTensionMomentumMorris(;
                                                       surface_tension_coefficient=0.072)
        fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation, surface_tension,
                                                   surface_normal_method=normal_method,
                                                   reference_particle_spacing=particle_spacing)

        # The wall continues the fluid lattice: the top wall row is one particle spacing
        # below the bottom fluid row.
        wall = RectangularShape(particle_spacing, (9, 3), (0.0, -0.3);
                                density=reference_density)
        boundary_model = BoundaryModelDummyParticles(wall; fluid_system,
                                                     boundary_density_calculator=AdamiPressureExtrapolation())
        boundary_system = WallBoundarySystem(wall, boundary_model)
        semi = Semidiscretization(fluid_system, boundary_system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

        return (; fluid, fluid_system, boundary_system, semi, v_ode, u_ode)
    end

    flat_pool = build_flat_pool(nothing)
    (; fluid, fluid_system, boundary_system, semi, v_ode, u_ode) = flat_pool
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

    # The continuous support moment identifies wall-completed bulk stencils as interior while
    # retaining the planar free surface. Both regions must have zero CSS acceleration.
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
                                   wall=true, walldistance=2.0, boundary_system_type=:wall)

    rigid_system, rigid_boundary, rigid_semi,
    rigid_ode = create_fluid_system(coordinates, velocity, mass, density, particle_spacing,
                                    SurfaceTensionMorris(surface_tension_coefficient=0.072);
                                    NDIMS, smoothing_length, smoothing_kernel,
                                    surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
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
