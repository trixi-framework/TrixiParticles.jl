"""
Validation-only corrected wetted-area contact energy used by the R4-W workbench.

The force cache lives in this mutable model so validation does not change production caches or the
public API.
"""
mutable struct CorrectedWettedAreaContact{T <: AbstractFloat} <:
               TrixiParticles.AbstractContactAngleModel
    contact_angle::T
    normalized_edge_shift::T
    flooded_reference::T
    explicit_force::Matrix{T}
    density_force::Matrix{T}
    wall_reaction::Matrix{T}
    wall_weight::Vector{T}
    density_conjugate::Vector{T}
    exposed::BitVector
    raw_area::T
    corrected_area::T
    area_derivative::T
    energy::T
    explicit_reaction_residual::T
    density_resultant::T
    total_momentum_residual::T
    max_explicit_reaction_residual::T
    max_density_resultant::T
    max_total_momentum_residual::T
    evaluations::Int
end

function CorrectedWettedAreaContact(contact_angle, normalized_edge_shift,
                                    flooded_reference)
    angle = TrixiParticles.validate_contact_angle(contact_angle)
    0 < angle < 180 ||
        throw(ArgumentError("R4-W requires `contact_angle` strictly inside (0, 180) degrees"))
    isfinite(normalized_edge_shift) ||
        throw(ArgumentError("`normalized_edge_shift` must be finite"))
    isfinite(flooded_reference) && flooded_reference > 0 ||
        throw(ArgumentError("`flooded_reference` must be finite and positive"))

    angle_, shift_,
    reference_ = promote(float(angle), float(normalized_edge_shift),
                         float(flooded_reference))
    T = typeof(angle_)
    return CorrectedWettedAreaContact{T}(angle_, shift_, reference_, zeros(T, 0, 0),
                                         zeros(T, 0, 0), zeros(T, 0, 0), T[], T[],
                                         BitVector(), zero(T), zero(T), zero(T), zero(T),
                                         zero(T), zero(T), zero(T), zero(T), zero(T),
                                         zero(T), 0)
end

@inline function r4_wetted_area_smoothstep(value)
    return value^2 * (3 - 2value)
end

@inline function r4_wetted_area_smoothstep_derivative(value)
    return 6value * (1 - value)
end

@inline function r4_contact_cosine(model::CorrectedWettedAreaContact)
    model.contact_angle == 90 && return zero(model.contact_angle)
    return cosd(model.contact_angle)
end

function TrixiParticles.convert_contact_model(model::CorrectedWettedAreaContact, ELTYPE)
    return CorrectedWettedAreaContact(convert(ELTYPE, model.contact_angle),
                                      convert(ELTYPE, model.normalized_edge_shift),
                                      convert(ELTYPE, model.flooded_reference))
end

function TrixiParticles.create_cache_surface_normal(surface_normal_method::TrixiParticles.ColorfieldSurfaceNormal{<:Any,
                                                                                                                  <:CorrectedWettedAreaContact},
                                                    ELTYPE, NDIMS, nparticles)
    cache = TrixiParticles.create_cache_surface_normal(TrixiParticles.ColorfieldSurfaceNormal(),
                                                       ELTYPE, NDIMS, nparticles)
    boundary_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    return (; cache..., boundary_normal)
end

function resize_r4_wetted_area_cache!(model, ndims, nfluid, nwall)
    if size(model.explicit_force) != (ndims, nfluid)
        model.explicit_force = zeros(eltype(model.explicit_force), ndims, nfluid)
        model.density_force = zeros(eltype(model.density_force), ndims, nfluid)
        model.density_conjugate = zeros(eltype(model.density_conjugate), nfluid)
    else
        fill!(model.explicit_force, 0)
        fill!(model.density_force, 0)
        fill!(model.density_conjugate, 0)
    end

    if size(model.wall_reaction) != (ndims, nwall)
        model.wall_reaction = zeros(eltype(model.wall_reaction), ndims, nwall)
        model.wall_weight = zeros(eltype(model.wall_weight), nwall)
        model.exposed = falses(nwall)
    else
        fill!(model.wall_reaction, 0)
        fill!(model.wall_weight, 0)
        fill!(model.exposed, false)
    end
    return model
end

function r4_force_resultant(forces)
    resultant = zeros(eltype(forces), size(forces, 1))
    scale = zero(eltype(forces))
    for particle in axes(forces, 2)
        force_norm2 = zero(eltype(forces))
        for dim in axes(forces, 1)
            value = forces[dim, particle]
            resultant[dim] += value
            force_norm2 += value^2
        end
        scale += sqrt(force_norm2)
    end
    return resultant, scale
end

@inline function r4_relative_residual(residual, scale)
    residual_norm = sqrt(sum(abs2, residual))
    iszero(scale) && return iszero(residual_norm) ? zero(scale) : oftype(scale, Inf)
    return residual_norm / scale
end

function r4_wetted_area_boundary_system(semi)
    boundary_system = nothing
    for candidate in semi.systems
        candidate isa TrixiParticles.AbstractBoundarySystem || continue
        hasproperty(candidate, :boundary_model) || continue
        haskey(candidate.boundary_model.cache, :colorfield) || continue
        isnothing(boundary_system) ||
            throw(ArgumentError("R4-W validation currently requires exactly one colorfield boundary"))
        boundary_system = candidate
    end
    isnothing(boundary_system) &&
        throw(ArgumentError("R4-W validation requires one colorfield boundary"))
    return boundary_system
end

function TrixiParticles.compute_contact_angle_cache!(system::TrixiParticles.AbstractFluidSystem,
                                                     surface_normal_method::TrixiParticles.ColorfieldSurfaceNormal{<:Any,
                                                                                                                    <:CorrectedWettedAreaContact},
                                                     v, u, v_ode, u_ode, semi)
    TrixiParticles.ndims(system) == 3 ||
        throw(ArgumentError("R4-W corrected wetted area is currently defined only in 3D"))

    model = surface_normal_method.contact_model
    boundary_system = r4_wetted_area_boundary_system(semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    coordinates = TrixiParticles.current_coordinates(u, system)
    boundary_coordinates = TrixiParticles.current_coordinates(u_boundary, boundary_system)
    nfluid = TrixiParticles.nparticles(system)
    nwall = TrixiParticles.nparticles(boundary_system)
    ndims = TrixiParticles.ndims(system)
    resize_r4_wetted_area_cache!(model, ndims, nfluid, nwall)

    particle_spacing = system.cache.reference_particle_spacing
    particle_area = particle_spacing^2
    smoothing_length = TrixiParticles.initial_smoothing_length(system)
    exposed_height = maximum(@view boundary_coordinates[3, :])
    height_tolerance = 10eps(abs(exposed_height) + particle_spacing)
    colorfield = boundary_system.boundary_model.cache.colorfield

    raw_area = zero(eltype(system))
    for boundary_particle in TrixiParticles.eachparticle(boundary_system)
        exposed = isapprox(boundary_coordinates[3, boundary_particle], exposed_height;
                           atol=height_tolerance)
        model.exposed[boundary_particle] = exposed
        exposed || continue
        fraction = clamp(colorfield[boundary_particle] / model.flooded_reference, 0, 1)
        raw_area += particle_area * r4_wetted_area_smoothstep(fraction)
    end

    raw_radius = sqrt(raw_area / pi)
    edge_shift = model.normalized_edge_shift * smoothing_length
    corrected_radius = max(raw_radius - edge_shift, zero(raw_radius))
    corrected_area = pi * corrected_radius^2
    area_derivative = raw_radius > eps(raw_radius) ? corrected_radius / raw_radius :
                      zero(raw_radius)
    for boundary_particle in TrixiParticles.eachparticle(boundary_system)
        model.exposed[boundary_particle] || continue
        fraction = colorfield[boundary_particle] / model.flooded_reference
        0 < fraction < 1 || continue
        model.wall_weight[boundary_particle] = area_derivative * particle_area /
                                               model.flooded_reference *
                                               r4_wetted_area_smoothstep_derivative(fraction)
    end

    model.raw_area = raw_area
    model.corrected_area = corrected_area
    model.area_derivative = area_derivative
    contact_cosine = r4_contact_cosine(model)
    surface_tension_coefficient = system.surface_tension.surface_tension_coefficient
    coefficient = surface_tension_coefficient * contact_cosine
    model.energy = iszero(contact_cosine) ? zero(contact_cosine) :
                   -coefficient * corrected_area

    if !iszero(coefficient)
        # Differentiate each exposed-wall color sample explicitly and retain its opposite
        # reaction. The kernel-value sum becomes the density conjugate q_a.
        TrixiParticles.foreach_point_neighbor(system, boundary_system, coordinates,
                                              boundary_coordinates, semi;
                                              points=TrixiParticles.each_integrated_particle(system),
                                              parallelization_backend=TrixiParticles.SerialBackend()) do particle,
                                                                                                         neighbor,
                                                                                                         pos_diff,
                                                                                                         distance
            wall_weight = model.wall_weight[neighbor]
            iszero(wall_weight) && return
            density = TrixiParticles.current_density(v, system, particle)
            mass = TrixiParticles.hydrodynamic_mass(system, particle)
            kernel_value = TrixiParticles.smoothing_kernel(system, distance, particle)
            gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                            particle)
            pair_force = coefficient * mass / density * wall_weight * gradient
            for dim in eachindex(pair_force)
                model.explicit_force[dim, particle] += pair_force[dim]
                model.wall_reaction[dim, neighbor] -= pair_force[dim]
            end
            model.density_conjugate[particle] += wall_weight * kernel_value
        end

        for particle in TrixiParticles.eachparticle(system)
            density = TrixiParticles.current_density(v, system, particle)
            model.density_conjugate[particle] *= coefficient / density^2
        end

        # Pair q_a with the ContinuityDensity pressure operator. Accumulating one force per
        # unordered pair makes the fluid-fluid resultant vanish by construction.
        TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                              semi;
                                              points=TrixiParticles.each_integrated_particle(system),
                                              parallelization_backend=TrixiParticles.SerialBackend()) do particle,
                                                                                                         neighbor,
                                                                                                         pos_diff,
                                                                                                         distance
            neighbor > particle || return
            mass_a = TrixiParticles.hydrodynamic_mass(system, particle)
            mass_b = TrixiParticles.hydrodynamic_mass(system, neighbor)
            density_a = TrixiParticles.current_density(v, system, particle)
            density_b = TrixiParticles.current_density(v, system, neighbor)
            gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                            particle)
            pair_coefficient = model.density_conjugate[particle] * density_a / density_b +
                               model.density_conjugate[neighbor] * density_b / density_a
            pair_force = -mass_a * mass_b * pair_coefficient * gradient
            for dim in eachindex(pair_force)
                model.density_force[dim, particle] += pair_force[dim]
                model.density_force[dim, neighbor] -= pair_force[dim]
            end
        end
    end

    explicit_resultant, explicit_scale = r4_force_resultant(model.explicit_force)
    density_resultant, density_scale = r4_force_resultant(model.density_force)
    wall_resultant, wall_scale = r4_force_resultant(model.wall_reaction)
    model.explicit_reaction_residual = r4_relative_residual(explicit_resultant +
                                                            wall_resultant,
                                                            explicit_scale + wall_scale)
    model.density_resultant = r4_relative_residual(density_resultant, density_scale)
    model.total_momentum_residual = r4_relative_residual(explicit_resultant +
                                                         density_resultant +
                                                         wall_resultant,
                                                         explicit_scale + density_scale +
                                                         wall_scale)
    model.max_explicit_reaction_residual = max(model.max_explicit_reaction_residual,
                                               model.explicit_reaction_residual)
    model.max_density_resultant = max(model.max_density_resultant,
                                      model.density_resultant)
    model.max_total_momentum_residual = max(model.max_total_momentum_residual,
                                            model.total_momentum_residual)
    model.evaluations += 1
    return system
end

@inline function TrixiParticles.contact_angle_acceleration(surface_tension::Union{TrixiParticles.SurfaceTensionMorris,
                                                                                  TrixiParticles.SurfaceTensionMomentumMorris},
                                                           particle_system,
                                                           surface_normal_method::TrixiParticles.ColorfieldSurfaceNormal{<:Any,
                                                                                                                         <:CorrectedWettedAreaContact},
                                                           particle, rho_a,
                                                           vector_template)
    model = surface_normal_method.contact_model
    explicit_force = TrixiParticles.extract_svector(model.explicit_force, particle_system,
                                                    particle)
    density_force = TrixiParticles.extract_svector(model.density_force, particle_system,
                                                   particle)
    mass = TrixiParticles.hydrodynamic_mass(particle_system, particle)
    return (explicit_force + density_force) / mass
end

corrected_wetted_area_contact_diagnostics(model) = nothing

function corrected_wetted_area_contact_diagnostics(model::CorrectedWettedAreaContact)
    explicit_resultant, explicit_scale = r4_force_resultant(model.explicit_force)
    density_resultant, density_scale = r4_force_resultant(model.density_force)
    wall_resultant, wall_scale = r4_force_resultant(model.wall_reaction)
    return (; energy=model.energy, raw_area=model.raw_area,
            corrected_area=model.corrected_area, area_derivative=model.area_derivative,
            explicit_resultant, density_resultant, wall_resultant,
            explicit_force_scale=explicit_scale, density_force_scale=density_scale,
            wall_force_scale=wall_scale,
            explicit_reaction_residual=model.explicit_reaction_residual,
            density_resultant_residual=model.density_resultant,
            total_momentum_residual=model.total_momentum_residual,
            max_explicit_reaction_residual=model.max_explicit_reaction_residual,
            max_density_resultant_residual=model.max_density_resultant,
            max_total_momentum_residual=model.max_total_momentum_residual,
            evaluations=model.evaluations, cache_bytes=Base.summarysize(model))
end
