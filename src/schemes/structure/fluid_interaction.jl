# Structure-fluid coupling shared by TLSPH and rigid-body systems.
# EDAC provides its own particle-local pressure average. Other fluid schemes keep
# their absolute pressures by subtracting zero in the physical pair operator.
@inline average_pressure(system, particle) = zero(eltype(system))

# Match the fluid scheme's force prefactors: EDAC/IISPH use unity, while WCSPH
# delegates to its configured free-surface correction for the two pair densities.
@inline interaction_force_correction(system, rho_a, rho_b) = (1, 1, 1)

@inline function interaction_force_correction(system::WeaklyCompressibleSPHSystem,
                                              rho_a, rho_b)
    return free_surface_correction(system_correction(system), system, rho_a, rho_b)
end

@inline function sum_interaction_contributions(a, b)
    # Reduce a vector momentum/force contribution and a scalar density rate together
    # without packing quantities with different meanings into one static vector.
    dv_a, drho_a = a
    dv_b, drho_b = b
    return dv_a + dv_b, drho_a + drho_b
end

# Evaluate the fluid particle's physical acceleration to assemble its structural
# reaction. Shifting transport terms are excluded from the interfacial traction.
# In this operator a is the fluid particle and b is its boundary/structure neighbor:
# pos_diff = x_f - x_s and grad_kernel is evaluated at f in that same direction.
# The return value is acceleration of f; the caller applies -m_f to obtain force on s.
@propagate_inbounds function physical_fluid_pair_acceleration(particle_system,
                                                              neighbor_system,
                                                              v_particle_system,
                                                              v_neighbor_system,
                                                              particle, neighbor,
                                                              m_a, m_b, p_a, p_b, rho_a,
                                                              rho_b,
                                                              v_a, v_b, pos_diff, distance,
                                                              sound_speed, grad_kernel,
                                                              correction)
    # EDAC subtracts the fluid particle's local mean from both pressures. Using the
    # boundary's own mean (or unreduced pressure) would change the fluid's pair force.
    p_avg = average_pressure(particle_system, particle)
    viscosity_correction, pressure_correction,
    surface_tension_correction = interaction_force_correction(particle_system, rho_a, rho_b)
    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                        particle, neighbor, m_a, m_b,
                                        p_a - p_avg, p_b - p_avg, rho_a, rho_b,
                                        pos_diff, distance, grad_kernel, correction)
    dv_particle = dv_pressure * pressure_correction
    # Keep systems, velocities, displacement, and gradient fluid-first as one unit.
    # The viscosity helper supplies model-specific ghost velocities where needed;
    # artificial viscosity must see the same approaching-particle test as the fluid RHS.
    dv_particle = add_dv_viscosity(dv_particle, particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance, sound_speed,
                                   m_a, m_b, rho_a, rho_b, v_a, v_b, grad_kernel,
                                   viscosity_correction)
    # Surface-force dispatch skips inapplicable models. In particular, rigid-body
    # adhesion uses the same fluid-frame displacement before the single reaction sign.
    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)
    dv_particle = add_dv_surface_tension(dv_particle, surface_tension_a, surface_tension_b,
                                         particle_system, neighbor_system, particle,
                                         neighbor,
                                         pos_diff, distance, rho_a, rho_b, grad_kernel,
                                         surface_tension_correction)
    return add_dv_adhesion(dv_particle, surface_tension_a,
                           particle_system, neighbor_system, particle, neighbor,
                           pos_diff, distance)
end

# Here the traversal labels are reversed: a=s (structure), b=f (fluid), and the
# neighbor search returns pos_diff=x_s-x_f. Physical operators below are evaluated
# with fluid-first arguments and r_fs=-pos_diff, then F_s=-m_f*a_f^physical.
# The default particle range advances free particles; callers such as mechanical
# work calculation can explicitly include clamped particles to recover their loads.
function interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   particle_system,
                                   neighbor_system::AbstractFluidSystem, semi;
                                   eachparticle=each_integrated_particle(particle_system))
    sound_speed = system_sound_speed(neighbor_system)
    correction = system_correction(neighbor_system)

    # Coupling follows the current spatial configuration, unlike TLSPH elastic
    # self-interaction, whose neighborhood is built in the initial configuration.
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    backend = semi.parallelization_backend

    # Match the fluid's support and its uniform h-relative squared-distance rule.
    # The support factor does not enter the near-zero criterion.
    compact_support_ = compact_support(neighbor_system, particle_system)
    h = initial_smoothing_length(neighbor_system)
    zero_distance_squared = eps(typeof(h)) * h^2

    # Each task owns one structural particle. Neighbor contributions remain local
    # until reduction, so force_per_particle and dv need no per-pair atomic updates.
    @threaded semi for particle in eachparticle
        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
        # corresponding to the rest density of the fluid and not the material density.
        # This mass/pressure/density is the boundary state used by the forward fluid
        # interaction. Material mass enters only when writing TLSPH acceleration.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)

        # Accumulate force and density rate separately, as in the fluid RHS.
        init = (zero(v_a), zero(rho_a))

        # Keep the returned name out of the closure to avoid allocations.
        (force_particle_,
         drho_particle_) = @inbounds mapreduce_neighbor(sum_interaction_contributions,
                                                        system_coords,
                                                        neighbor_coords,
                                                        neighborhood_search, backend,
                                                        particle;
                                                        init) do particle,
                                                                 neighbor,
                                                                 pos_diff,
                                                                 distance
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
            v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

            # Corrected coincident-particle gradients can be finite and nonzero.
            # Apply the relative rule only when this correction skips those pairs.
            if distance > compact_support_ ||
               (skip_zero_distance(neighbor_system) && distance^2 < zero_distance_squared)
                return init
            end

            drho_particle = @inbounds add_continuity_equation(zero(rho_a),
                                                              particle_system,
                                                              neighbor_system,
                                                              particle, neighbor, pos_diff,
                                                              distance, m_b, rho_a, rho_b,
                                                              v_a, v_b)

            # Evaluate grad_f W(r_fs) directly at the fluid particle. Neighborhood
            # corrections contain particle-local terms, so -grad_f W(r_sf) need not
            # equal grad_f W(-r_sf). Negating a structure-oriented corrected gradient
            # would therefore give the wrong pressure/viscous reaction.
            grad_kernel_fluid = smoothing_kernel_grad_unsafe(neighbor_system, -pos_diff,
                                                             distance, neighbor)

            # Read fluid pressure first, then use the same boundary-pressure dispatch
            # as the forward interaction. Pair-local mirroring, where supported,
            # needs p_f even though the traversal visits the structure particle first.
            p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)
            p_a = @inbounds neighbor_pressure(v_particle_system, particle_system, particle,
                                              p_b)

            # Fluid-first ordering preserves the actual fluid force, including the
            # approaching-particle condition of artificial viscosity.
            dv_fluid = @inbounds physical_fluid_pair_acceleration(neighbor_system,
                                                                  particle_system,
                                                                  v_neighbor_system,
                                                                  v_particle_system,
                                                                  neighbor, particle,
                                                                  m_b, m_a, p_b, p_a, rho_b,
                                                                  rho_a,
                                                                  v_b, v_a, -pos_diff,
                                                                  distance,
                                                                  sound_speed,
                                                                  grad_kernel_fluid,
                                                                  correction)

            # m_b is fluid mass in this traversal: Newton's third law transfers the
            # opposite physical force, not merely an acceleration with reversed sign.
            return -m_b * dv_fluid, drho_particle
        end

        # TLS divides the accumulated force by material mass; rigid bodies retain
        # particle forces for the later resultant/torque reduction. Boundary density
        # is a separate scalar state derivative and is written through its own dispatch.
        @inbounds write_fluid_force!(dv, particle_system, force_particle_, particle)
        @inbounds write_drho_particle!(dv, particle_system, drho_particle_,
                                       particle)
    end

    return dv
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::AbstractStructureSystem,
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b)
    # Pressure-extrapolated and summation-density boundaries have no density ODE
    # contribution; only the ContinuityDensity specialization below evolves rho_s.
    return drho_particle
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                                TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b)
    # Continuity has structure-first orientation: (v_s-v_f) dot grad_s W. Its
    # gradient uses the hydrodynamic boundary kernel/correction, independently of
    # the fluid's gradient and the TLSPH elastic self-interaction kernel.
    # The fluid density formulation selects the weight: m_f for SummationDensity,
    # or (rho_s/rho_f)*m_f for ContinuityDensity.
    grad_kernel = hydrodynamic_kernel_grad(particle_system, pos_diff, distance, particle)

    return add_continuity_equation(drho_particle,
                                   density_calculator(neighbor_system),
                                   m_b, rho_a, rho_b, v_a, v_b, grad_kernel, particle)
end

@inline function write_drho_particle!(dv, ::AbstractSystem, drho_particle, particle)
    # A boundary model without an integrated density must not modify a velocity row.
    return dv
end

@propagate_inbounds function write_drho_particle!(dv,
                                                  ::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                          TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                                  drho_particle, particle)
    # ContinuityDensity appends hydrodynamic density to v; momentum occupies the
    # preceding rows. Add rather than overwrite when several fluid systems contribute.
    dv[end, particle] += drho_particle

    return dv
end
