using .FSIPairFixtures: structure_fluid_pair_state

@testset verbose=true "Unequal structure-fluid kernel supports" begin
    function pair_forces(state)
        # Evaluate both directions independently; an absent reverse neighbor must
        # fail conservation even when the fluid acceleration itself is nonzero.
        (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
        dv_fluid, dv_structure = zero(v_fluid), zero(v_structure)
        TrixiParticles.reset_interaction_caches!(structure)
        TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                 fluid, structure, semi)
        TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid, u_fluid,
                                 structure, fluid, semi)
        force = structure isa RigidBodySystem ? copy(structure.force_per_particle[:, 1]) :
                structure.mass[1] * dv_structure[1:2, 1]
        @test force ≈ -1100.0 * dv_fluid[1:2, 1]
        return force, dv_structure
    end

    # Fluid support is 2. Boundary h=0.5,1,2 gives support 1,2,4. At r=1.5,
    # the smallest boundary search used to omit a pair still seen by the fluid.
    # Check both NHS sharing strategies so radius reuse cannot hide that omission.
    for kind in (:tlsph, :rigid), handler in (PairsNHSHandler, SharedNHSHandler),
        h in (0.5, 1.0, 2.0)
        state = structure_fluid_pair_state(; structure_kind=kind,
                                           boundary_smoothing_length=h,
                                           neighborhood_search_handler=handler)
        force, _ = pair_forces(state)
        # Uncorrected continuity-density pressure force uses p_f+p_s=500+230=730.
        # The fluid gradient stays fixed as the boundary support changes.
        expected = 1100.0 * 700.0 * 730.0 / (1005.0 * 950.0) * [15 / (56pi), 0.0]
        @test force ≈ expected
        @test TrixiParticles.compact_support(state.structure, state.fluid) == max(2h, 2.0)
    end

    # Precomputed neighbor lists must also be initialized with the enlarged support.
    for kind in (:tlsph, :rigid)
        state = structure_fluid_pair_state(; structure_kind=kind,
                                           boundary_smoothing_length=0.5,
                                           neighborhood_search=PrecomputedNeighborhoodSearch{2}(),
                                           neighborhood_search_handler=PairsNHSHandler)
        force, _ = pair_forces(state)
        @test !iszero(force)
    end

    # Density survives fluid pair skips, but never extends beyond its own kernel.
    # The cases isolate: fluid near-zero skip with finite boundary gradient;
    # a pair outside fluid support but inside boundary support; and the reverse.
    # At h=1, r=2e-8 lies between the boundary's ~1.49e-8 and WCSPH's ~2.98e-8 cutoffs.
    for (h, distance) in ((1.0, 2.0e-8), (2.0, 2.5), (0.5, 1.5))
        state = structure_fluid_pair_state(; structure_kind=:rigid, distance,
                                           boundary_density=ContinuityDensity(),
                                           boundary_smoothing_length=h)
        force, dv_structure = pair_forces(state)
        q = distance / h
        # Differentiate the 2D cubic spline directly: normalization/h = 10/(7pi*h^3).
        derivative = q >= 2 ? 0.0 :
                     (q < 1 ? -3q + 2.25q^2 : -0.75 * (2 - q)^2) * 10 / (7pi * h^3)
        # The pair lies on the x-axis, so only v_sx-v_fx=-0.75 enters boundary continuity.
        @test dv_structure[end, 1] ≈ 950.0 / 1005.0 * 1100.0 * (-0.75) * derivative
        @test iszero(force) == (distance < 3.0e-8 || distance > 2)
    end
end
