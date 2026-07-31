function sample_foot_pocket(particle_spacing, center, blade)
    _, foot_pocket = _sample_foot_pocket(particle_spacing, center, blade)

    # Move the fin to the center of the tank.
    foot_pocket.coordinates .+= center
    blade.coordinates .+= center

    return foot_pocket
end

function _sample_foot_pocket(particle_spacing, center, blade)
    file = joinpath(examples_dir(), "preprocessing", "data", "hyper_bifins_x.dxf")
    geometry = load_geometry(file)

    point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                        winding_number_factor=0.4,
                                                        hierarchical_winding=true)

    # Returns an `InitialCondition`.
    # Note that the density is arbitrary, as density and mass will be overwritten later.
    shape_sampled = ComplexShape(geometry; particle_spacing, density=1000.0,
                                 grid_offset=(0.0, particle_spacing / 2),
                                 point_in_geometry_algorithm)

    # Make sure that the coordinates use FP64, even when `particle_spacing` is FP32.
    coordinates_fp64 = Float64.(shape_sampled.coordinates)
    shape_sampled = TrixiParticles.@set shape_sampled.coordinates = coordinates_fp64

    # When particles are too close together, keep the ones from `beam`
    # instead of `packed_foot` to ensure that the blade doesn't have holes.
    foot_pocket = setdiff(shape_sampled, blade)

    return geometry, foot_pocket
end

function sample_and_pack(particle_spacing, center, blade, fluid)
    geometry, foot_pocket = _sample_foot_pocket(particle_spacing, center, blade)

    foot_sdf = SignedDistanceField(geometry, particle_spacing;
                                   max_signed_distance=4 * particle_spacing,
                                   use_for_boundary_packing=true)

    boundary_for_packing = sample_boundary(foot_sdf; boundary_density=1000.0,
                                           boundary_thickness=4 * particle_spacing)

    # Make sure that the coordinates use FP64, even when `particle_spacing` is FP32.
    coordinates_fp64 = Float64.(boundary_for_packing.coordinates)
    boundary_for_packing = TrixiParticles.@set boundary_for_packing.coordinates = coordinates_fp64
    boundary_for_packing = setdiff(boundary_for_packing, blade)

    background_pressure = 1.0
    smoothing_length_packing = 0.8 * particle_spacing
    foot_packing_system = ParticlePackingSystem(foot_pocket;
                                                smoothing_length=smoothing_length_packing,
                                                signed_distance_field=foot_sdf,
                                                background_pressure)

    # This is just a thin layer of fluid particles against which the foot is packed.
    fluid_packing_system = ParticlePackingSystem(boundary_for_packing;
                                                 smoothing_length=smoothing_length_packing,
                                                 signed_distance_field=foot_sdf,
                                                 is_boundary=true, background_pressure,
                                                 boundary_compress_factor=0.8)

    blade_packing_system = ParticlePackingSystem(blade;
                                                 smoothing_length=smoothing_length_packing,
                                                 fixed_system=true,
                                                 signed_distance_field=nothing,
                                                 background_pressure)

    semi_packing = Semidiscretization(foot_packing_system, fluid_packing_system,
                                      blade_packing_system)

    ode_packing = semidiscretize(semi_packing, (0.0, 100.0))

    sol_packing = solve(ode_packing, RDPK3SpFSAL35();
                        save_everystep=false,
                        callback=CallbackSet(InfoCallback(interval=50), UpdateCallback()),
                        abstol=1e-8, dtmax=1e-1)

    packed_foot = InitialCondition(sol_packing, foot_packing_system, semi_packing)

    # Move the fin to the center of the tank
    packed_foot.coordinates .+= center
    blade.coordinates .+= center

    # When particles are too close together, keep the ones from `blade`
    # instead of `packed_foot` to ensure that the blade doesn't have holes.
    foot_pocket = setdiff(packed_foot, blade)

    # ======================================================================================
    # ==== Pack the fluid against the fin
    fin = union(blade, foot_pocket)
    fluid = setdiff(fluid, fin)

    # Only pack the fluid in a small window around the fin to reduce the number of particles.
    pack_window = TrixiParticles.Polygon(stack([
                                                center .+ [-0.4, -0.08],
                                                center .+ [-0.2, -0.08],
                                                center .+ [-0.06, -0.02],
                                                center .+ [0.62, -0.02],
                                                center .+ [0.62, 0.02],
                                                center .+ [0.05, 0.02],
                                                center .+ [-0.1, 0.1],
                                                center .+ [-0.4, 0.1],
                                                center .+ [-0.4, -0.08]
                                            ]))

    # Extract the particles that fall inside this window
    pack_fluid = intersect(fluid, pack_window)
    # and those outside the window.
    fixed_fluid = setdiff(fluid, pack_fluid)
    fixed_union = union(fixed_fluid, fin)

    fluid_packing_system = ParticlePackingSystem(pack_fluid;
                                                 smoothing_length=smoothing_length_packing,
                                                 signed_distance_field=nothing,
                                                 background_pressure)

    fixed_packing_system = ParticlePackingSystem(fixed_union;
                                                 smoothing_length=smoothing_length_packing,
                                                 fixed_system=true,
                                                 signed_distance_field=nothing,
                                                 background_pressure)

    min_corner = minimum(fixed_union.coordinates, dims=2)
    max_corner = maximum(fixed_union.coordinates, dims=2)
    cell_list = FullGridCellList(; min_corner, max_corner)
    neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                    update_strategy=ParallelUpdate())

    semi_packing = Semidiscretization(fluid_packing_system, fixed_packing_system;
                                      neighborhood_search)

    ode_packing = semidiscretize(semi_packing, (0.0, 50.0))

    sol_packing = solve(ode_packing, RDPK3SpFSAL35();
                        save_everystep=false,
                        callback=CallbackSet(InfoCallback(interval=50), UpdateCallback()),
                        abstol=1e-8, dtmax=1e-1)

    fluid = InitialCondition(sol_packing, fluid_packing_system, semi_packing)
    fluid = union(fluid, fixed_fluid)

    return foot_pocket, fluid
end
