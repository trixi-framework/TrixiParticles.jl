# Hydrodynamic sums of the dummy-particle boundary models of fluid-coupled structures.
# The correction values and, with `SummationDensity`, the density of the boundary particles
# are sums over the hydrodynamic neighborhood in the current configuration, i.e., over all
# particles of fluids, boundaries and structures within the compact support of the boundary
# model, including the structure itself. This neighborhood is independent of the frozen
# neighborhood of the TLSPH solid mechanics and of the contact search of rigid bodies.
@inline has_hydrodynamic_neighborhood(system) = false

@inline function has_hydrodynamic_neighborhood(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                             RigidBodySystem{<:BoundaryModelDummyParticles}})
    (; correction, density_calculator) = system.boundary_model
    return density_calculator isa SummationDensity ||
           correction isa Union{ShepardKernelCorrection, KernelCorrection,
                 GradientCorrection, BlendedGradientCorrection,
                 MixedKernelGradientCorrection}
end

@inline function hydrodynamic_support(system)
    model = system.boundary_model
    return compact_support(model.smoothing_kernel, initial_smoothing_length(model))
end

# Structures without boundary model, e.g., rigid bodies with only a contact model, have no
# hydrodynamic density or mass. They must not contribute to the hydrodynamic sums of the
# boundary model, even though other interactions like mechanical contact remain enabled.
@inline function has_hydrodynamic_interaction(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                            RigidBodySystem{<:BoundaryModelDummyParticles}},
                                              neighbor::Union{TotalLagrangianSPHSystem{Nothing},
                                                              RigidBodySystem{Nothing}},
                                              semi)
    return false
end

@inline function get_summation_neighborhood_search(model::BoundaryModelDummyParticles,
                                                   system::Union{TotalLagrangianSPHSystem,
                                                                 RigidBodySystem},
                                                   neighbor, semi)
    return get_hydrodynamic_neighborhood_search(system, neighbor, semi)
end

# The regular searches of structures with fluid systems already use the compact support of
# the boundary model and are updated in the current configuration (see `update_nhs!`),
# so they are reused for the hydrodynamic sums instead of creating and updating a copy.
@inline uses_regular_hydrodynamic_search(neighbor) = false
@inline uses_regular_hydrodynamic_search(neighbor::AbstractFluidSystem) = true

# Only update the neighbor coordinates of hydrodynamic searches if they change over time
@inline hydrodynamic_points_moving(neighbor) = true
@inline hydrodynamic_points_moving(neighbor::WallBoundarySystem) = neighbor.ismoving[]

# Structure density summation runs in its own global phase, independently of the isolated
# TLSPH updates used by `SplitIntegrationCallback`.
@inline update_structure_boundary_density!(system, v, u, v_ode, u_ode, semi) = system

function update_structure_boundary_density!(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                          RigidBodySystem{<:BoundaryModelDummyParticles}},
                                            v, u, v_ode, u_ode, semi)
    boundary_model = system.boundary_model
    # Shepard's raw numerator is assembled with its coefficient and applied only after
    # every system has completed its density-correction moments.
    correction_density(boundary_model.correction) isa ShepardKernelCorrection &&
        return system
    update_structure_boundary_density!(boundary_model, system, v, u, v_ode, u_ode, semi)
    return system
end

@inline function update_structure_boundary_density!(boundary_model, system, v, u,
                                                    v_ode, u_ode, semi)
    return boundary_model
end

function update_structure_boundary_density!(boundary_model::BoundaryModelDummyParticles{SummationDensity},
                                            system, v, u, v_ode, u_ode, semi)
    return update_density!(boundary_model, system, v, u, v_ode, u_ode, semi)
end
