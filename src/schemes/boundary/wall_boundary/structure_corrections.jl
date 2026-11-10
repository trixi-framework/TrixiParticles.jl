# Hydrodynamic dummy-particle correction caches for fluid-coupled structures.
# These use the boundary model independently of elastic self-interaction.
@inline has_boundary_correction(system) = false

@inline function has_boundary_correction(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                       RigidBodySystem{<:BoundaryModelDummyParticles}})
    correction = system.boundary_model.correction
    return correction_density(correction) !== nothing ||
           correction_gradient(correction) !== nothing
end

@inline function hydrodynamic_correction(system::RigidBodySystem{<:BoundaryModelDummyParticles})
    return correction_gradient(system.boundary_model.correction)
end

@inline function hydrodynamic_smoothing_kernel_grad(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                                 RigidBodySystem{<:BoundaryModelDummyParticles}},
                                                   pos_diff, distance, particle)
    return smoothing_kernel_grad(system.boundary_model, pos_diff, distance, particle)
end

function update_density_correction_values!(system::RigidBodySystem{<:BoundaryModelDummyParticles},
                                           v, u, v_ode, u_ode, semi, t)
    update_density_correction_values!(system.boundary_model, system, v, u, v_ode, u_ode,
                                      semi)
    return system
end

function update_density_correction!(system::RigidBodySystem{<:BoundaryModelDummyParticles},
                                     v, u, v_ode, u_ode, semi, t)
    update_density_correction!(system.boundary_model, system, v, u, v_ode, u_ode, semi)
    return system
end

function update_gradient_correction!(system::RigidBodySystem{<:BoundaryModelDummyParticles},
                                      v, u, v_ode, u_ode, semi, t)
    update_gradient_correction!(system.boundary_model, system, v, u, v_ode, u_ode, semi)
    return system
end

@inline function boundary_correction_support(system)
    model = system.boundary_model
    return compact_support(model.smoothing_kernel, initial_smoothing_length(model))
end

# Mechanical contact can be enabled without hydrodynamic density or mass on the neighbor.
# Such bodies must not contribute to the boundary-model correction moments.
@inline function has_correction_interaction(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                          RigidBodySystem{<:BoundaryModelDummyParticles}},
                                            neighbor::RigidBodySystem{Nothing}, semi)
    return false
end

@inline function get_correction_neighborhood_search(model::BoundaryModelDummyParticles,
                                                    system::Union{TotalLagrangianSPHSystem,
                                                                  RigidBodySystem},
                                                    neighbor, semi)
    return get_boundary_correction_neighborhood_search(system, neighbor, semi)
end
