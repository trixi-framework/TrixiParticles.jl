using LinearAlgebra
using OrdinaryDiffEqLowStorageRK
using Random
using Statistics
using TrixiParticles

function cube_surface_mesh(side_length, bottom_height)
    half_width = side_length / 2
    vertices = [SVector(-half_width, -half_width, bottom_height),
        SVector(half_width, -half_width, bottom_height),
        SVector(half_width, half_width, bottom_height),
        SVector(-half_width, half_width, bottom_height)]
    face_vertices = [(vertices[1], vertices[2], vertices[3]),
        (vertices[1], vertices[3], vertices[4])]
    face_normals = fill(SVector(0.0, 0.0, 1.0), 2)
    bottom = TrixiParticles.TriangleMesh(face_vertices, face_normals, vertices)

    return extrude_geometry(bottom, side_length)
end

function jittered_initial_condition(initial_condition; relative_amplitude, seed)
    coordinates = copy(initial_condition.coordinates)
    rng = MersenneTwister(seed)
    offsets = 2 .* rand(rng, size(coordinates)...) .- 1
    offsets .-= mean(offsets; dims=2)
    coordinates .+= relative_amplitude * initial_condition.particle_spacing .* offsets

    return InitialCondition(; coordinates,
                            velocity=copy(initial_condition.velocity),
                            mass=copy(initial_condition.mass),
                            density=copy(initial_condition.density),
                            pressure=copy(initial_condition.pressure),
                            particle_spacing=initial_condition.particle_spacing)
end

function packed_cube_initial_condition(; particle_spacing, cube_side_length=0.01,
                                       cube_bottom_height=0.0025,
                                       density=1000.0, relative_jitter=0.1,
                                       seed=20260805, maxiters=1000)
    n_cube = ntuple(_ -> round(Int, cube_side_length / particle_spacing), 3)
    cube_min = (-cube_side_length / 2, -cube_side_length / 2,
                cube_bottom_height)
    lattice = RectangularShape(particle_spacing, n_cube, cube_min; density)
    jittered = jittered_initial_condition(lattice;
                                          relative_amplitude=relative_jitter, seed)

    geometry = cube_surface_mesh(cube_side_length, cube_bottom_height)
    boundary_thickness = 3 * particle_spacing
    signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                                use_for_boundary_packing=true,
                                                max_signed_distance=boundary_thickness)
    boundary = sample_boundary(signed_distance_field; boundary_density=density,
                               boundary_thickness, place_on_shell=false)

    smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
    smoothing_length = 0.8 * particle_spacing
    background_pressure = 1.0
    packing_system = ParticlePackingSystem(jittered; smoothing_kernel,
                                           smoothing_length,
                                           signed_distance_field,
                                           background_pressure)
    boundary_system = ParticlePackingSystem(boundary; smoothing_kernel,
                                            smoothing_length, is_boundary=true,
                                            boundary_compress_factor=0.8,
                                            signed_distance_field,
                                            boundary_thickness,
                                            background_pressure)
    semi = Semidiscretization(packing_system, boundary_system)
    ode = semidiscretize(semi, (0.0, 10_000.0))
    solution = solve(ode, RDPK3SpFSAL35(); abstol=1.0e-7, reltol=1.0e-4,
                     save_everystep=false, maxiters,
                     callback=CallbackSet(UpdateCallback()))
    packed = InitialCondition(solution, packing_system, semi)
    nparticles(packed) == nparticles(lattice) ||
        error("particle packing removed $(nparticles(lattice) - nparticles(packed)) particles")

    displacement = packed.coordinates - lattice.coordinates
    rms_displacement = sqrt(sum(abs2, displacement) / nparticles(lattice)) /
                       particle_spacing
    maximum_displacement = maximum(norm, eachcol(displacement)) / particle_spacing
    diagnostics = (; accepted_steps=solution.stats.naccept,
                   rejected_steps=solution.stats.nreject,
                   rms_displacement, maximum_displacement,
                   relative_jitter, seed, maxiters)
    return packed, diagnostics
end
