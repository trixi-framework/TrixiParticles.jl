# Hydrodynamic dummy-particle corrections for fluid-coupled structures. These use
# the boundary-model context independently of TLSPH's elastic self-interaction.
@inline function hydrodynamic_kernel_grad(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                        RigidBodySystem{<:BoundaryModelDummyParticles}},
                                          pos_diff, distance, particle)
    return smoothing_kernel_grad(system.boundary_model, pos_diff, distance, particle)
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
