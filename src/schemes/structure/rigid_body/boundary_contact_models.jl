abstract type AbstractRigidBoundaryContactModel end

struct RigidBoundaryContactModel{ELTYPE <: Real} <: AbstractRigidBoundaryContactModel
    normal_stiffness::ELTYPE
    normal_damping::ELTYPE
    static_friction_coefficient::ELTYPE
    kinetic_friction_coefficient::ELTYPE
    tangential_stiffness::ELTYPE
    tangential_damping::ELTYPE
    contact_distance::ELTYPE
    stick_velocity_tolerance::ELTYPE
    penetration_slop::ELTYPE
    torque_free::Bool
    resting_contact_projection::Bool
end

function RigidBoundaryContactModel(; normal_stiffness,
                                   normal_damping=0.0,
                                   static_friction_coefficient=0.5,
                                   kinetic_friction_coefficient=0.4,
                                   tangential_stiffness=0.0,
                                   tangential_damping=0.0,
                                   contact_distance=0.0,
                                   stick_velocity_tolerance=1e-6,
                                   penetration_slop=0.0,
                                   torque_free=false,
                                   resting_contact_projection=true)
    ELTYPE = promote_type(typeof(normal_stiffness),
                          typeof(normal_damping),
                          typeof(static_friction_coefficient),
                          typeof(kinetic_friction_coefficient),
                          typeof(tangential_stiffness),
                          typeof(tangential_damping),
                          typeof(contact_distance),
                          typeof(stick_velocity_tolerance),
                          typeof(penetration_slop))

    normal_stiffness_ = convert(ELTYPE, normal_stiffness)
    normal_damping_ = convert(ELTYPE, normal_damping)
    static_friction_coefficient_ = convert(ELTYPE, static_friction_coefficient)
    kinetic_friction_coefficient_ = convert(ELTYPE, kinetic_friction_coefficient)
    tangential_stiffness_ = convert(ELTYPE, tangential_stiffness)
    tangential_damping_ = convert(ELTYPE, tangential_damping)
    contact_distance_ = convert(ELTYPE, contact_distance)
    stick_velocity_tolerance_ = convert(ELTYPE, stick_velocity_tolerance)
    penetration_slop_ = convert(ELTYPE, penetration_slop)

    normal_stiffness_ > 0 ||
        throw(ArgumentError("`normal_stiffness` must be positive"))
    normal_damping_ >= 0 ||
        throw(ArgumentError("`normal_damping` must be non-negative"))
    static_friction_coefficient_ >= 0 ||
        throw(ArgumentError("`static_friction_coefficient` must be non-negative"))
    kinetic_friction_coefficient_ >= 0 ||
        throw(ArgumentError("`kinetic_friction_coefficient` must be non-negative"))
    kinetic_friction_coefficient_ <= static_friction_coefficient_ ||
        throw(ArgumentError("`kinetic_friction_coefficient` must be <= `static_friction_coefficient`"))
    tangential_stiffness_ >= 0 ||
        throw(ArgumentError("`tangential_stiffness` must be non-negative"))
    tangential_damping_ >= 0 ||
        throw(ArgumentError("`tangential_damping` must be non-negative"))
    contact_distance_ >= 0 ||
        throw(ArgumentError("`contact_distance` must be non-negative"))
    stick_velocity_tolerance_ >= 0 ||
        throw(ArgumentError("`stick_velocity_tolerance` must be non-negative"))
    penetration_slop_ >= 0 ||
        throw(ArgumentError("`penetration_slop` must be non-negative"))

    return RigidBoundaryContactModel(normal_stiffness_, normal_damping_,
                                     static_friction_coefficient_,
                                     kinetic_friction_coefficient_,
                                     tangential_stiffness_, tangential_damping_,
                                     contact_distance_, stick_velocity_tolerance_,
                                     penetration_slop_, Bool(torque_free),
                                     Bool(resting_contact_projection))
end

"""
    PerfectElasticBoundaryContactModel(; normal_stiffness,
                                       contact_distance=0.0,
                                       contact_distance_factor=1.0,
                                       stick_velocity_tolerance=1e-8,
                                       torque_free=true)

Specification model for idealized rigid-wall elastic impacts.
It is converted to a [`RigidBoundaryContactModel`](@ref) inside
`RigidBodySystem` via constructor-based adaptation
`RigidBoundaryContactModel(model, particle_spacing, ELTYPE)`.
"""
struct PerfectElasticBoundaryContactModel{ELTYPE <: Real} <: AbstractRigidBoundaryContactModel
    normal_stiffness::ELTYPE
    contact_distance::ELTYPE
    contact_distance_factor::ELTYPE
    stick_velocity_tolerance::ELTYPE
    torque_free::Bool
end

function PerfectElasticBoundaryContactModel(; normal_stiffness,
                                            contact_distance=0.0,
                                            contact_distance_factor=1.0,
                                            stick_velocity_tolerance=1e-8,
                                            torque_free=true)
    ELTYPE = promote_type(typeof(normal_stiffness),
                          typeof(contact_distance),
                          typeof(contact_distance_factor),
                          typeof(stick_velocity_tolerance))

    normal_stiffness_ = convert(ELTYPE, normal_stiffness)
    contact_distance_ = convert(ELTYPE, contact_distance)
    contact_distance_factor_ = convert(ELTYPE, contact_distance_factor)
    stick_velocity_tolerance_ = convert(ELTYPE, stick_velocity_tolerance)

    normal_stiffness_ > 0 ||
        throw(ArgumentError("`normal_stiffness` must be positive"))
    contact_distance_ >= 0 ||
        throw(ArgumentError("`contact_distance` must be non-negative"))
    contact_distance_factor_ > 0 ||
        throw(ArgumentError("`contact_distance_factor` must be positive"))
    stick_velocity_tolerance_ >= 0 ||
        throw(ArgumentError("`stick_velocity_tolerance` must be non-negative"))

    return PerfectElasticBoundaryContactModel(normal_stiffness_, contact_distance_,
                                              contact_distance_factor_,
                                              stick_velocity_tolerance_,
                                              Bool(torque_free))
end

@inline contact_shear_modulus(youngs_modulus, poisson_ratio) = youngs_modulus / (2.0 *
                                                                                  (1.0 +
                                                                                   poisson_ratio))

@inline function contact_effective_youngs_modulus(material, wall_material)
    return 1.0 / ((1.0 - material.poisson_ratio^2) / material.youngs_modulus +
                  (1.0 - wall_material.poisson_ratio^2) / wall_material.youngs_modulus)
end

@inline function contact_effective_shear_modulus(material, wall_material)
    shear_material = contact_shear_modulus(material.youngs_modulus, material.poisson_ratio)
    shear_wall = contact_shear_modulus(wall_material.youngs_modulus,
                                       wall_material.poisson_ratio)

    return 1.0 / ((2.0 - material.poisson_ratio) / shear_material +
                  (2.0 - wall_material.poisson_ratio) / shear_wall)
end

@inline function damping_ratio_from_restitution(restitution)
    restitution_clamped = clamp(restitution, 0.0, 1.0)
    restitution_clamped >= 1.0 && return zero(restitution_clamped)
    restitution_clamped <= eps(restitution_clamped) && return one(restitution_clamped)

    log_restitution = log(restitution_clamped)
    return -log_restitution / sqrt(pi^2 + log_restitution^2)
end

# Restitution of the clamped Kelvin-Voigt contact used in `normal_contact_force_components`.
# This solves a dimensionless 1D impact problem where the spring-dashpot force is clamped
# to avoid attraction during decompression.
function clamped_restitution_from_damping_ratio(damping_ratio;
                                                dt=5.0e-4,
                                                max_time=60.0)
    zeta = max(Float64(damping_ratio), 0.0)
    zeta <= eps(zeta) && return 1.0

    penetration = 0.0
    penetration_rate = 1.0
    started = false
    time = 0.0

    while time < max_time
        function rhs(penetration_local, penetration_rate_local)
            force = penetration_local > 0 ? max(penetration_local +
                                                2.0 * zeta * penetration_rate_local,
                                                0.0) : 0.0
            return penetration_rate_local, -force
        end

        k1_p, k1_v = rhs(penetration, penetration_rate)
        k2_p, k2_v = rhs(penetration + 0.5 * dt * k1_p,
                         penetration_rate + 0.5 * dt * k1_v)
        k3_p, k3_v = rhs(penetration + 0.5 * dt * k2_p,
                         penetration_rate + 0.5 * dt * k2_v)
        k4_p, k4_v = rhs(penetration + dt * k3_p,
                         penetration_rate + dt * k3_v)

        penetration_next = penetration + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p +
                                                       k4_p)
        penetration_rate_next = penetration_rate + (dt / 6.0) * (k1_v + 2.0 * k2_v +
                                                                  2.0 * k3_v + k4_v)

        started |= penetration > 0.0
        if started && penetration > 0.0 && penetration_next <= 0.0 &&
           penetration_rate_next < 0.0
            theta = penetration / (penetration - penetration_next)
            rebound_rate = penetration_rate +
                           theta * (penetration_rate_next - penetration_rate)

            return max(0.0, -rebound_rate)
        end

        penetration = penetration_next
        penetration_rate = penetration_rate_next
        time += dt
    end

    return 0.0
end

# Invert `clamped_restitution_from_damping_ratio` via bisection.
function damping_ratio_from_restitution_clamped(restitution; tolerance=1.0e-8)
    target_restitution = clamp(Float64(restitution), 0.0, 1.0)
    target_restitution >= 1.0 && return 0.0
    target_restitution <= eps(target_restitution) && return 100.0

    lower = 0.0
    upper = 1.0
    restitution_upper = clamped_restitution_from_damping_ratio(upper)

    while restitution_upper > target_restitution && upper < 100.0
        lower = upper
        upper *= 2.0
        restitution_upper = clamped_restitution_from_damping_ratio(upper)
    end

    restitution_upper > target_restitution && return upper

    for _ in 1:70
        mid = 0.5 * (lower + upper)
        restitution_mid = clamped_restitution_from_damping_ratio(mid)

        if restitution_mid > target_restitution
            lower = mid
        else
            upper = mid
        end

        (upper - lower) <= tolerance * max(1.0, upper) && break
    end

    return 0.5 * (lower + upper)
end

@inline function default_body_mass(::Val{2}, density, radius)
    return density * pi * radius^2
end

@inline function default_body_mass(::Val{3}, density, radius)
    return density * (4.0 / 3.0) * pi * radius^3
end

@inline function drop_impact_velocity(center, gravity, radius, particle_spacing)
    isnothing(center) &&
        throw(ArgumentError("provide `impact_velocity` or `center` + `gravity` + `particle_spacing`"))
    isnothing(gravity) &&
        throw(ArgumentError("provide `impact_velocity` or `center` + `gravity` + `particle_spacing`"))
    isnothing(particle_spacing) &&
        throw(ArgumentError("`particle_spacing` is required when inferring `impact_velocity` from drop height"))
    particle_spacing > 0 ||
        throw(ArgumentError("`particle_spacing` must be positive"))

    gravity_magnitude = gravity isa Number ? abs(gravity) : norm(gravity)
    drop_height = max(center[end] - radius, particle_spacing)
    return sqrt(2.0 * gravity_magnitude * drop_height)
end

@inline function resolve_impact_velocity(impact_velocity, center, gravity, radius,
                                         particle_spacing)
    return isnothing(impact_velocity) ?
           drop_impact_velocity(center, gravity, radius, particle_spacing) :
           abs(impact_velocity)
end

function validate_linearized_hertz_mindlin_inputs(; effective_youngs_modulus,
                                                  effective_shear_modulus,
                                                  radius,
                                                  impact_velocity,
                                                  body_mass,
                                                  restitution,
                                                  static_friction_coefficient,
                                                  kinetic_friction_coefficient,
                                                  contact_distance,
                                                  contact_distance_factor,
                                                  stick_velocity_tolerance,
                                                  stick_velocity_tolerance_factor,
                                                  minimum_stick_velocity_tolerance,
                                                  penetration_slop)
    effective_youngs_modulus > 0 ||
        throw(ArgumentError("`effective_youngs_modulus` must be positive"))
    effective_shear_modulus > 0 ||
        throw(ArgumentError("`effective_shear_modulus` must be positive"))
    radius > 0 || throw(ArgumentError("`radius` must be positive"))
    impact_velocity >= 0 || throw(ArgumentError("`impact_velocity` must be non-negative"))
    body_mass > 0 || throw(ArgumentError("`body_mass` must be positive"))
    0 <= restitution <= 1 ||
        throw(ArgumentError("`restitution` must be in [0, 1]"))
    static_friction_coefficient >= 0 ||
        throw(ArgumentError("`static_friction_coefficient` must be non-negative"))
    kinetic_friction_coefficient >= 0 ||
        throw(ArgumentError("`kinetic_friction_coefficient` must be non-negative"))
    kinetic_friction_coefficient <= static_friction_coefficient ||
        throw(ArgumentError("`kinetic_friction_coefficient` must be <= `static_friction_coefficient`"))
    contact_distance >= 0 ||
        throw(ArgumentError("`contact_distance` must be non-negative"))
    contact_distance_factor > 0 ||
        throw(ArgumentError("`contact_distance_factor` must be positive"))
    stick_velocity_tolerance >= 0 ||
        throw(ArgumentError("`stick_velocity_tolerance` must be non-negative"))
    stick_velocity_tolerance_factor >= 0 ||
        throw(ArgumentError("`stick_velocity_tolerance_factor` must be non-negative"))
    minimum_stick_velocity_tolerance >= 0 ||
        throw(ArgumentError("`minimum_stick_velocity_tolerance` must be non-negative"))
    penetration_slop >= 0 ||
        throw(ArgumentError("`penetration_slop` must be non-negative"))
end

"""
    LinearizedHertzMindlinBoundaryContactModel(effective_youngs_modulus,
                                               effective_shear_modulus,
                                               radius, impact_velocity, body_mass,
                                               restitution,
                                               static_friction_coefficient;
                                               kinetic_friction_coefficient=0.9 * static_friction_coefficient,
                                               contact_distance=0.0,
                                               contact_distance_factor=2.0,
                                               stick_velocity_tolerance=0.0,
                                               stick_velocity_tolerance_factor=0.01,
                                               minimum_stick_velocity_tolerance=1e-5,
                                               penetration_slop=0.0,
                                               torque_free=false,
                                               resting_contact_projection=true)

Specification model for linearized Hertz-Mindlin rigid-wall contact.
It is converted to a [`RigidBoundaryContactModel`](@ref) inside
`RigidBodySystem` via constructor-based adaptation
`RigidBoundaryContactModel(model, particle_spacing, ELTYPE)`.

A convenience constructor is also provided:
`LinearizedHertzMindlinBoundaryContactModel(; material, wall_material, radius, ndims, ...)`.
"""
struct LinearizedHertzMindlinBoundaryContactModel{ELTYPE <: Real} <:
       AbstractRigidBoundaryContactModel
    effective_youngs_modulus::ELTYPE
    effective_shear_modulus::ELTYPE
    radius::ELTYPE
    impact_velocity::ELTYPE
    body_mass::ELTYPE
    restitution::ELTYPE
    static_friction_coefficient::ELTYPE
    kinetic_friction_coefficient::ELTYPE
    contact_distance::ELTYPE
    contact_distance_factor::ELTYPE
    stick_velocity_tolerance::ELTYPE
    stick_velocity_tolerance_factor::ELTYPE
    minimum_stick_velocity_tolerance::ELTYPE
    penetration_slop::ELTYPE
    torque_free::Bool
    resting_contact_projection::Bool
end

function LinearizedHertzMindlinBoundaryContactModel(effective_youngs_modulus,
                                                    effective_shear_modulus,
                                                    radius, impact_velocity, body_mass,
                                                    restitution,
                                                    static_friction_coefficient;
                                                    kinetic_friction_coefficient=0.9 *
                                                                                 static_friction_coefficient,
                                                    contact_distance=0.0,
                                                    contact_distance_factor=2.0,
                                                    stick_velocity_tolerance=0.0,
                                                    stick_velocity_tolerance_factor=0.01,
                                                    minimum_stick_velocity_tolerance=1e-5,
                                                    penetration_slop=0.0,
                                                    torque_free=false,
                                                    resting_contact_projection=true)
    ELTYPE = promote_type(typeof(effective_youngs_modulus),
                          typeof(effective_shear_modulus),
                          typeof(radius),
                          typeof(impact_velocity),
                          typeof(body_mass),
                          typeof(restitution),
                          typeof(static_friction_coefficient),
                          typeof(kinetic_friction_coefficient),
                          typeof(contact_distance),
                          typeof(contact_distance_factor),
                          typeof(stick_velocity_tolerance),
                          typeof(stick_velocity_tolerance_factor),
                          typeof(minimum_stick_velocity_tolerance),
                          typeof(penetration_slop))

    effective_youngs_modulus_ = convert(ELTYPE, effective_youngs_modulus)
    effective_shear_modulus_ = convert(ELTYPE, effective_shear_modulus)
    radius_ = convert(ELTYPE, radius)
    impact_velocity_ = convert(ELTYPE, impact_velocity)
    body_mass_ = convert(ELTYPE, body_mass)
    restitution_ = convert(ELTYPE, restitution)
    static_friction_coefficient_ = convert(ELTYPE, static_friction_coefficient)
    kinetic_friction_coefficient_ = convert(ELTYPE, kinetic_friction_coefficient)
    contact_distance_ = convert(ELTYPE, contact_distance)
    contact_distance_factor_ = convert(ELTYPE, contact_distance_factor)
    stick_velocity_tolerance_ = convert(ELTYPE, stick_velocity_tolerance)
    stick_velocity_tolerance_factor_ = convert(ELTYPE, stick_velocity_tolerance_factor)
    minimum_stick_velocity_tolerance_ = convert(ELTYPE, minimum_stick_velocity_tolerance)
    penetration_slop_ = convert(ELTYPE, penetration_slop)

    validate_linearized_hertz_mindlin_inputs(; effective_youngs_modulus=effective_youngs_modulus_,
                                             effective_shear_modulus=effective_shear_modulus_,
                                             radius=radius_,
                                             impact_velocity=impact_velocity_,
                                             body_mass=body_mass_,
                                             restitution=restitution_,
                                             static_friction_coefficient=static_friction_coefficient_,
                                             kinetic_friction_coefficient=kinetic_friction_coefficient_,
                                             contact_distance=contact_distance_,
                                             contact_distance_factor=contact_distance_factor_,
                                             stick_velocity_tolerance=stick_velocity_tolerance_,
                                             stick_velocity_tolerance_factor=stick_velocity_tolerance_factor_,
                                             minimum_stick_velocity_tolerance=minimum_stick_velocity_tolerance_,
                                             penetration_slop=penetration_slop_)

    return LinearizedHertzMindlinBoundaryContactModel(effective_youngs_modulus_,
                                                      effective_shear_modulus_,
                                                      radius_, impact_velocity_,
                                                      body_mass_, restitution_,
                                                      static_friction_coefficient_,
                                                      kinetic_friction_coefficient_,
                                                      contact_distance_,
                                                      contact_distance_factor_,
                                                      stick_velocity_tolerance_,
                                                      stick_velocity_tolerance_factor_,
                                                      minimum_stick_velocity_tolerance_,
                                                      penetration_slop_,
                                                      Bool(torque_free),
                                                      Bool(resting_contact_projection))
end

function LinearizedHertzMindlinBoundaryContactModel(; material, wall_material, radius,
                                                    ndims,
                                                    impact_velocity=nothing,
                                                    center=nothing, gravity=nothing,
                                                    particle_spacing=nothing,
                                                    body_mass=nothing,
                                                    restitution=nothing,
                                                    static_friction_coefficient=nothing,
                                                    kinetic_friction_coefficient=nothing,
                                                    contact_distance=0.0,
                                                    contact_distance_factor=2.0,
                                                    stick_velocity_tolerance=0.0,
                                                    stick_velocity_tolerance_factor=0.01,
                                                    minimum_stick_velocity_tolerance=1e-5,
                                                    penetration_slop=0.0,
                                                    torque_free=false,
                                                    resting_contact_projection=true)
    ndims == 2 || ndims == 3 ||
        throw(ArgumentError("`ndims` must be 2 or 3"))

    impact_velocity_ = resolve_impact_velocity(impact_velocity, center, gravity,
                                               radius, particle_spacing)
    if isnothing(body_mass)
        hasproperty(material, :density) ||
            throw(ArgumentError("`material.density` is required when `body_mass` is not provided"))
    end
    body_mass_ = isnothing(body_mass) ?
                 default_body_mass(Val(ndims), material.density, radius) :
                 body_mass
    restitution_ = isnothing(restitution) ? material.restitution : restitution
    static_friction_coefficient_ = isnothing(static_friction_coefficient) ?
                                   material.friction_coefficient :
                                   static_friction_coefficient
    kinetic_friction_coefficient_ = isnothing(kinetic_friction_coefficient) ?
                                    0.9 * static_friction_coefficient_ :
                                    kinetic_friction_coefficient

    return LinearizedHertzMindlinBoundaryContactModel(contact_effective_youngs_modulus(material,
                                                                                        wall_material),
                                                      contact_effective_shear_modulus(material,
                                                                                      wall_material),
                                                      radius,
                                                      impact_velocity_,
                                                      body_mass_,
                                                      restitution_,
                                                      static_friction_coefficient_;
                                                      kinetic_friction_coefficient=kinetic_friction_coefficient_,
                                                      contact_distance,
                                                      contact_distance_factor,
                                                      stick_velocity_tolerance,
                                                      stick_velocity_tolerance_factor,
                                                      minimum_stick_velocity_tolerance,
                                                      penetration_slop,
                                                      torque_free,
                                                      resting_contact_projection)
end

function RigidBoundaryContactModel(model::PerfectElasticBoundaryContactModel,
                                   particle_spacing, ::Type{ELTYPE}) where {ELTYPE}
    particle_spacing_ = convert(ELTYPE, particle_spacing)
    particle_spacing_ > 0 ||
        throw(ArgumentError("`particle_spacing` must be positive"))

    contact_distance = model.contact_distance > 0 ? convert(ELTYPE, model.contact_distance) :
                       convert(ELTYPE, model.contact_distance_factor) * particle_spacing_

    return RigidBoundaryContactModel(; normal_stiffness=convert(ELTYPE,
                                                                model.normal_stiffness),
                                     normal_damping=zero(ELTYPE),
                                     static_friction_coefficient=zero(ELTYPE),
                                     kinetic_friction_coefficient=zero(ELTYPE),
                                     tangential_stiffness=zero(ELTYPE),
                                     tangential_damping=zero(ELTYPE),
                                     contact_distance,
                                     stick_velocity_tolerance=convert(ELTYPE,
                                                                      model.stick_velocity_tolerance),
                                     penetration_slop=zero(ELTYPE),
                                     torque_free=model.torque_free,
                                     resting_contact_projection=false)
end

function RigidBoundaryContactModel(model::LinearizedHertzMindlinBoundaryContactModel,
                                   particle_spacing, ::Type{ELTYPE}) where {ELTYPE}
    particle_spacing_ = convert(ELTYPE, particle_spacing)
    particle_spacing_ > 0 ||
        throw(ArgumentError("`particle_spacing` must be positive"))

    effective_radius = convert(ELTYPE, model.radius)
    effective_youngs_modulus = convert(ELTYPE, model.effective_youngs_modulus)
    effective_shear_modulus = convert(ELTYPE, model.effective_shear_modulus)
    impact_velocity = convert(ELTYPE, model.impact_velocity)
    body_mass = convert(ELTYPE, model.body_mass)

    hertz_coefficient = convert(ELTYPE, 4.0 / 3.0) * effective_youngs_modulus *
                        sqrt(effective_radius)
    reference_penetration = ((convert(ELTYPE, 5.0 / 4.0) * body_mass * impact_velocity^2) /
                             hertz_coefficient)^convert(ELTYPE, 2.0 / 5.0)
    reference_penetration = max(reference_penetration,
                                convert(ELTYPE, 0.01) * particle_spacing_)

    normal_stiffness = convert(ELTYPE, 3.0 / 2.0) * hertz_coefficient *
                       sqrt(reference_penetration)
    contact_radius = sqrt(effective_radius * reference_penetration)
    tangential_stiffness = convert(ELTYPE, 8.0) * effective_shear_modulus *
                           contact_radius

    damping_ratio = convert(ELTYPE,
                            damping_ratio_from_restitution_clamped(model.restitution))
    normal_damping = convert(ELTYPE, 2.0) * damping_ratio *
                     sqrt(normal_stiffness * body_mass)
    tangential_damping = convert(ELTYPE, 2.0) * damping_ratio *
                         sqrt(tangential_stiffness * body_mass)

    static_friction_coefficient = convert(ELTYPE, model.static_friction_coefficient)
    kinetic_friction_coefficient = convert(ELTYPE, model.kinetic_friction_coefficient)

    if static_friction_coefficient <= eps(ELTYPE)
        tangential_stiffness = zero(ELTYPE)
        tangential_damping = zero(ELTYPE)
    end

    contact_distance = model.contact_distance > 0 ? convert(ELTYPE, model.contact_distance) :
                       convert(ELTYPE, model.contact_distance_factor) * particle_spacing_
    stick_velocity_tolerance = model.stick_velocity_tolerance > 0 ?
                               convert(ELTYPE, model.stick_velocity_tolerance) :
                               max(convert(ELTYPE,
                                           model.minimum_stick_velocity_tolerance),
                                   convert(ELTYPE,
                                           model.stick_velocity_tolerance_factor) *
                                   impact_velocity)

    return RigidBoundaryContactModel(; normal_stiffness,
                                     normal_damping,
                                     static_friction_coefficient,
                                     kinetic_friction_coefficient,
                                     tangential_stiffness,
                                     tangential_damping,
                                     contact_distance,
                                     stick_velocity_tolerance,
                                     penetration_slop=convert(ELTYPE,
                                                              model.penetration_slop),
                                     torque_free=model.torque_free,
                                     resting_contact_projection=model.resting_contact_projection)
end

function RigidBoundaryContactModel(model::RigidBoundaryContactModel, particle_spacing,
                                   ::Type{ELTYPE}) where {ELTYPE}
    contact_distance = model.contact_distance > 0 ? model.contact_distance :
                       convert(ELTYPE, particle_spacing)

    return RigidBoundaryContactModel(; normal_stiffness=convert(ELTYPE,
                                                                model.normal_stiffness),
                                     normal_damping=convert(ELTYPE, model.normal_damping),
                                     static_friction_coefficient=convert(ELTYPE,
                                                                         model.static_friction_coefficient),
                                     kinetic_friction_coefficient=convert(ELTYPE,
                                                                          model.kinetic_friction_coefficient),
                                     tangential_stiffness=convert(ELTYPE,
                                                                  model.tangential_stiffness),
                                     tangential_damping=convert(ELTYPE,
                                                                model.tangential_damping),
                                     contact_distance,
                                     stick_velocity_tolerance=convert(ELTYPE,
                                                                      model.stick_velocity_tolerance),
                                     penetration_slop=convert(ELTYPE,
                                                              model.penetration_slop),
                                     torque_free=model.torque_free,
                                     resting_contact_projection=model.resting_contact_projection)
end

create_contact_tangential_displacement(::Nothing, ELTYPE, ::Val{NDIMS}) where {NDIMS} = nothing

function create_contact_tangential_displacement(::RigidBoundaryContactModel, ELTYPE,
                                                ::Val{NDIMS}) where {NDIMS}
    return Dict{NTuple{3, Int}, SVector{NDIMS, ELTYPE}}()
end


function Base.show(io::IO, model::RigidBoundaryContactModel)
    @nospecialize model # reduce precompilation time

    print(io, "RigidBoundaryContactModel(")
    print(io, "normal_stiffness=", model.normal_stiffness)
    print(io, ", normal_damping=", model.normal_damping)
    print(io, ", mu_s=", model.static_friction_coefficient)
    print(io, ", mu_k=", model.kinetic_friction_coefficient)
    print(io, ", tangential_stiffness=", model.tangential_stiffness)
    print(io, ", tangential_damping=", model.tangential_damping)
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ", stick_velocity_tolerance=", model.stick_velocity_tolerance)
    print(io, ", penetration_slop=", model.penetration_slop)
    print(io, ", torque_free=", model.torque_free)
    print(io, ", resting_contact_projection=", model.resting_contact_projection)
    print(io, ")")
end

function Base.show(io::IO, model::PerfectElasticBoundaryContactModel)
    @nospecialize model # reduce precompilation time

    print(io, "PerfectElasticBoundaryContactModel(")
    print(io, "normal_stiffness=", model.normal_stiffness)
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ", contact_distance_factor=", model.contact_distance_factor)
    print(io, ", stick_velocity_tolerance=", model.stick_velocity_tolerance)
    print(io, ", torque_free=", model.torque_free)
    print(io, ")")
end

function Base.show(io::IO, model::LinearizedHertzMindlinBoundaryContactModel)
    @nospecialize model # reduce precompilation time

    print(io, "LinearizedHertzMindlinBoundaryContactModel(")
    print(io, "E*=", model.effective_youngs_modulus)
    print(io, ", G*=", model.effective_shear_modulus)
    print(io, ", radius=", model.radius)
    print(io, ", impact_velocity=", model.impact_velocity)
    print(io, ", body_mass=", model.body_mass)
    print(io, ", restitution=", model.restitution)
    print(io, ", mu_s=", model.static_friction_coefficient)
    print(io, ", mu_k=", model.kinetic_friction_coefficient)
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ", contact_distance_factor=", model.contact_distance_factor)
    print(io, ", stick_velocity_tolerance=", model.stick_velocity_tolerance)
    print(io, ", stick_velocity_tolerance_factor=", model.stick_velocity_tolerance_factor)
    print(io, ", minimum_stick_velocity_tolerance=", model.minimum_stick_velocity_tolerance)
    print(io, ", penetration_slop=", model.penetration_slop)
    print(io, ", torque_free=", model.torque_free)
    print(io, ", resting_contact_projection=", model.resting_contact_projection)
    print(io, ")")
end
