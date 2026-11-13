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

create_cache_structure_boundary(boundary_model, initial_condition) = (;)

function create_cache_structure_boundary(boundary_model::BoundaryModelDummyParticles{SummationDensity},
                                         initial_condition)
    if correction_density(boundary_model.correction) isa ShepardKernelCorrection
        # Keep the numerator separate until the global coefficient phase is complete so
        # other systems never observe a partially updated structure density.
        return (; boundary_density_numerator=similar(boundary_model.cache.density))
    end

    return (;)
end

@inline function hydrodynamic_smoothing_kernel_grad(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                                  RigidBodySystem{<:BoundaryModelDummyParticles}},
                                                    pos_diff, distance, particle)
    return smoothing_kernel_grad(system.boundary_model, pos_diff, distance, particle)
end

function update_density_correction_values!(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                         RigidBodySystem{<:BoundaryModelDummyParticles}},
                                           v, u, v_ode, u_ode, semi, t)
    density_correction = correction_density(system.boundary_model.correction)
    compute_structure_density_correction_values!(system.boundary_model, system,
                                                 density_correction, u, v_ode, u_ode, semi)

    return system
end

@inline function compute_structure_density_correction_values!(boundary_model, system,
                                                              density_correction, u,
                                                              v_ode, u_ode, semi)
    return compute_boundary_correction_values!(boundary_model, system, density_correction,
                                               u, v_ode, u_ode, semi)
end

function compute_structure_density_correction_values!(boundary_model::BoundaryModelDummyParticles{SummationDensity},
                                                      system,
                                                      ::ShepardKernelCorrection, u,
                                                      v_ode, u_ode, semi)
    # Assemble the raw summation numerator together with the Shepard coefficient to avoid
    # a second neighbor traversal. Both use the boundary model's hydrodynamic neighborhood.
    return compute_shepard_coeff!(system, current_coordinates(u, system), v_ode, u_ode,
                                  semi,
                                  boundary_model.cache.kernel_correction_coefficient,
                                  system.cache.boundary_density_numerator;
                                  kernel_data=boundary_model)
end

function update_density_correction!(system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                  RigidBodySystem{<:BoundaryModelDummyParticles}},
                                    v, u, v_ode, u_ode, semi, t)
    density_correction = correction_density(system.boundary_model.correction)
    apply_structure_density_correction!(system.boundary_model, system, density_correction,
                                        v, u, v_ode, u_ode, semi)

    return system
end

@inline function apply_structure_density_correction!(boundary_model, system,
                                                     density_correction,
                                                     v, u, v_ode, u_ode, semi)
    return update_density_correction!(boundary_model, system, v, u, v_ode, u_ode, semi)
end

function apply_structure_density_correction!(boundary_model::BoundaryModelDummyParticles{SummationDensity},
                                             system, ::ShepardKernelCorrection,
                                             v, u, v_ode, u_ode, semi)
    (; density, kernel_correction_coefficient) = boundary_model.cache
    density_numerator = system.cache.boundary_density_numerator

    @threaded semi for particle in eachparticle(system)
        @inbounds density[particle] = density_numerator[particle] /
                                      kernel_correction_coefficient[particle]
    end

    return boundary_model
end

function update_gradient_correction!(system::RigidBodySystem{<:BoundaryModelDummyParticles},
                                     v, u, v_ode, u_ode, semi, t)
    update_gradient_correction!(system.boundary_model, system, v, u, v_ode, u_ode, semi)
    return system
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
