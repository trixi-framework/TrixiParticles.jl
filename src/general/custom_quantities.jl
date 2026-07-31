"""
    kinetic_energy

Returns the total kinetic energy of all particles in a system.
"""
function kinetic_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)

    velocity = reinterpret(reshape, SVector{ndims(system), eltype(v)},
                           view(current_velocity(v, system), :,
                                each_active_particle(system)))
    mass = view(system.mass, each_active_particle(system))

    return mapreduce(+, velocity, mass) do v_i, m_i
        return m_i * dot(v_i, v_i) / 2
    end
end

function kinetic_energy(system::AbstractStructureSystem,
                        dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    mass = system.mass
    energy = zero(eltype(system))

    return sum(each_active_particle(system)) do particle
        v_i = current_velocity(v, system, particle)
        energy += mass[particle] * dot(v_i, v_i) / 2
    end
end

function kinetic_energy(system::AbstractBoundarySystem,
                        dv_ode, du_ode, v_ode, u_ode, semi, t)
    return zero(eltype(system))
end

"""
    total_mass

Returns the total mass of all particles in a system.
"""
function total_mass(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return sum(system.mass)
end

function total_mass(system::AbstractBoundarySystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    # It does not make sense to return a mass for boundary systems.
    # The material density and therefore the physical mass of the boundary is not relevant
    # when simulating a solid, stationary wall. The boundary always behaves as if it had
    # infinite mass. There is no momentum transferred to the boundary on impact.
    #
    # When the dummy particles model is used, i.e., boundary particles behave like fluid
    # particles when interacting with actual fluid particles, the boundary particles do have
    # a "hydrodynamic mass", which corresponds to the fluid density, but this is only
    # relevant for the fluid interaction, and it has no connection to the physical mass
    # of the boundary. Returning the "hydrodynamic mass" here would thus be misleading.
    return NaN
end

"""
    max_pressure

Returns the maximum pressure over all particles in a system.
"""
function max_pressure(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return maximum(current_pressure(v, system))
end

function max_pressure(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    min_pressure

Returns the minimum pressure over all particles in a system.
"""
function min_pressure(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return minimum(current_pressure(v, system))
end

function min_pressure(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    avg_pressure

Returns the average pressure over all particles in a system.
"""
function avg_pressure(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    sum_ = sum(current_pressure(v, system))
    return sum_ / nparticles(system)
end

function avg_pressure(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    max_density

Returns the maximum density over all particles in a system.
"""
function max_density(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return maximum(current_density(v, system))
end

function max_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    min_density

Returns the minimum density over all particles in a system.
"""
function min_density(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return minimum(current_density(v, system))
end

function min_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    avg_density

Returns the average_density over all particles in a system.
"""
function avg_density(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    sum_ = sum(current_density(v, system))
    return sum_ / nparticles(system)
end

function avg_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

mutable struct TLSPHMotionCalculator{P}
    initialized           :: Bool
    system_index          :: Int
    position              :: P
    neighboring_particles :: Vector{Int}
    weights               :: Vector{Float64}
    sum_weights           :: Float64
end

function tlsph_motion(system, semi, position)
    if !isa(system, TotalLagrangianSPHSystem)
        throw(ArgumentError("TLSPH motion can only be computed for TLSPH systems."))
    end
    system_index = system_indices(system, semi)
    position_ = SVector(Tuple(position))

    return TLSPHMotionCalculator(false, system_index, position_, Int[], Float64[], 0.0)
end

function reset!(calculator::TLSPHMotionCalculator)
    calculator.initialized = false
end

function (calculator::TLSPHMotionCalculator)(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    (; position, neighboring_particles, weights, sum_weights) = calculator

    if system_indices(system, semi) != calculator.system_index
        return nothing
    end

    if !calculator.initialized
        initialize!(calculator, system)
    end

    displacement = zero(position)
    deformation_grad = zero(SMatrix{ndims(system), ndims(system), eltype(system)})

    for i in eachindex(neighboring_particles)
        particle = neighboring_particles[i]
        weight = weights[i]

        displacement += weight * (current_coords(system, particle) -
                                  initial_coords(system, particle))
        deformation_grad += weight * deformation_gradient(system, particle)
    end

    displacement /= sum_weights
    deformation_grad /= sum_weights

    # In 2D, this is the angle of the proper orthogonal factor in the polar
    # decomposition of the deformation gradient.
    rotation = atan(deformation_grad[2, 1] - deformation_grad[1, 2],
                    deformation_grad[1, 1] + deformation_grad[2, 2])

    return displacement, rotation
end

function initialize!(calculator::TLSPHMotionCalculator, system)
    (; position) = calculator

    search_radius = compact_support(system, system)

    # The reference configuration is fixed, so determine the kernel support only once.
    neighboring_particles = findall(eachparticle(system)) do particle
        initial_position = initial_coords(system, particle)
        pos_diff = initial_position - position
        distance2 = dot(pos_diff, pos_diff)

        return distance2 <= search_radius^2
    end

    @assert !isempty(neighboring_particles)

    weights = map(neighboring_particles) do particle
        initial_position = initial_coords(system, particle)
        pos_diff = initial_position - position
        distance = sqrt(dot(pos_diff, pos_diff))
        kernel_weight = kernel(system.smoothing_kernel, distance, system.smoothing_length)

        volume = system.mass[particle] / system.material_density[particle]
        return volume * kernel_weight
    end

    sum_weights = sum(weights)
    @assert sum_weights > eps(sum_weights)

    calculator.initialized = true
    calculator.neighboring_particles = neighboring_particles
    calculator.weights = weights
    calculator.sum_weights = sum_weights
end
