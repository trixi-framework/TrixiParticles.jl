# Hydrodynamic dummy-particle correction caches for fluid-coupled structures.
# These use the boundary model independently of elastic self-interaction.
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
