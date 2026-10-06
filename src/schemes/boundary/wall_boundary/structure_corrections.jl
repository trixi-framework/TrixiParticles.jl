# Hydrodynamic dummy-particle corrections for fluid-coupled structures. These use
# the boundary-model context independently of TLSPH's elastic self-interaction.
@inline function pressure_acceleration(particle_system,
                                       neighbor_system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                              RigidBodySystem{<:BoundaryModelDummyParticles}},
                                       particle, neighbor, m_a, m_b, p_a, p_b, rho_a, rho_b,
                                       pos_diff, distance, W_a,
                                       correction::Union{KernelCorrection,
                                                         GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection})
    # The boundary model supplies the kernel and correction state for fluid pressure.
    return pressure_acceleration(particle_system, neighbor_system.boundary_model,
                                 particle, neighbor, m_a, m_b, p_a, p_b, rho_a, rho_b,
                                 pos_diff, distance, W_a, correction)
end

@inline has_boundary_correction(system) = false

@inline function has_boundary_correction(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                       RigidBodySystem{<:BoundaryModelDummyParticles}})
    correction = system.boundary_model.correction
    return correction isa Union{ShepardKernelCorrection, KernelCorrection,
                 GradientCorrection, BlendedGradientCorrection,
                 MixedKernelGradientCorrection}
end

@inline function boundary_correction_support(system)
    model = system.boundary_model
    return compact_support(model.smoothing_kernel, initial_smoothing_length(model))
end

@inline function get_correction_neighborhood_search(model::BoundaryModelDummyParticles,
                                                    system::Union{TotalLagrangianSPHSystem,
                                                                  RigidBodySystem},
                                                    neighbor, semi)
    return get_boundary_correction_neighborhood_search(system, neighbor, semi)
end

function compute_correction_values!(system::Union{TotalLagrangianSPHSystem,
                                                  RigidBodySystem},
                                    ::ShepardKernelCorrection, u, v_ode, u_ode, semi)
    model = system.boundary_model
    return compute_shepard_coeff!(system, current_coordinates(u, system), v_ode, u_ode,
                                  semi, model.cache.kernel_correction_coefficient;
                                  kernel_system=model)
end

function compute_correction_values!(system::Union{TotalLagrangianSPHSystem,
                                                  RigidBodySystem},
                                    correction::Union{KernelCorrection,
                                                      MixedKernelGradientCorrection},
                                    u, v_ode, u_ode, semi)
    model = system.boundary_model
    return compute_correction_values!(system, correction, current_coordinates(u, system),
                                      v_ode, u_ode, semi,
                                      model.cache.kernel_correction_coefficient,
                                      model.cache.dw_gamma; kernel_system=model)
end

function compute_gradient_correction_matrix!(correction::Union{GradientCorrection,
                                                               BlendedGradientCorrection,
                                                               MixedKernelGradientCorrection},
                                             model::BoundaryModelDummyParticles,
                                             system::Union{TotalLagrangianSPHSystem,
                                                           RigidBodySystem},
                                             u, v_ode, u_ode, semi)
    return compute_gradient_correction_matrix!(model.cache.correction_matrix, system,
                                               current_coordinates(u, system), v_ode, u_ode,
                                               semi, correction, model.smoothing_kernel;
                                               kernel_system=model)
end
