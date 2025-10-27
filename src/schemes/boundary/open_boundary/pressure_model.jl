"""
    RCRWindkesselModel(; characteristic_resistance, peripheral_resistance, compliance)

The `RCRWindkessel` model is a biomechanical lumped-parameter representation
that captures the relationship between pressure and flow in pulsatile systems, e.g., vascular systems.
It is derived from an electrical circuit analogy and consists of three elements:

- characteristic restistance (``R_1``): Represents the proximal resistance at the vessel entrance.
  It models the immediate pressure drop that arises at the entrance of a vessel segment,
  due either to a geometric narrowing or to a mismatch in characteristic impedance between adjacent segments.
  A larger ``R_1`` produces a sharper initial pressure rise at the onset of flow.
- peripheral restistance (``R_2``): Represents the distal resistance,
  which controls the sustained outflow into the peripheral circulation and thereby determines the level of the mean pressure.
  A high ``R_2`` maintains a higher pressure (reduced outflow), whereas a low ``R_2`` allows a faster pressure decay.
- compliance (``C``): Connected in parallel with ``R_2`` and represents the capacity of elastic walls
  to store and release volume; in other words, it models the "stretchiness" of walls.
  Analogous to a capacitor in an electrical circuit, it absorbs blood when pressure rises and releases it during diastole.
  The presence of ``C`` smooths pulsatile flow and produces a more uniform outflow profile.

 Lumped-parameter models for the vascular system are well described in the literature (e.g. [Westerhof2008](@cite)).
 A practical step-by-step procedure for identifying the corresponding model parameters is provided by [Gasser2021](@cite).

# Keywords
- `characteristic_resistance`: characteristic restistance (``R_1``)
- `peripheral_resistance`: peripheral restistance (``R_2``)
- `compliance`: compliance (``C``)
"""
struct RCRWindkesselModel{ELTYPE <: Real}
    characteristic_resistance :: ELTYPE
    peripheral_resistance     :: ELTYPE
    compliance                :: ELTYPE
    is_prescribed             :: Bool
end

function RCRWindkesselModel(; characteristic_resistance, peripheral_resistance, compliance)
    return RCRWindkesselModel(characteristic_resistance, peripheral_resistance, compliance,
                              true)
end

function Base.show(io::IO, ::MIME"text/plain", pressure_model::RCRWindkesselModel)
    @nospecialize pressure_model # reduce precompilation time

    if get(io, :compact, false)
        show(io, pressure_model)
    else
        summary_header(io, "RCRWindkesselModel")
        summary_line(io, "characteristic_resistance",
                     pressure_model.characteristic_resistance)
        summary_line(io, "peripheral_resistance",
                     pressure_model.peripheral_resistance)
        summary_line(io, "compliance", pressure_model.compliance)
        summary_footer(io)
    end
end

function update_pressure_model!(system::OpenBoundarySystem, v, u, semi, dt)
    isnothing(system.pressure_model_values) && return system

    calculate_flow_rate_and_pressure!(system, v, u, dt)

    return system
end

function calculate_flow_rate_and_pressure!(system, v, u, dt)
    for (zone_id, boundary_zone) in enumerate(system.boundary_zones)
        if boundary_zone.pressure_model.is_prescribed
            calculate_flow_rate_and_pressure!(boundary_zone.pressure_model, system,
                                              boundary_zone, zone_id, v, u, dt)
        end
    end

    return system
end

function calculate_flow_rate_and_pressure!(pressure_model, system, boundary_zone,
                                           zone_id, v, u, dt)
    dt < sqrt(eps()) && return pressure_model
    (; particle_spacing) = system.initial_condition
    (; characteristic_resistance, peripheral_resistance, compliance) = pressure_model
    (; flow_rate, pressure) = system.pressure_model_values[zone_id]
    (; face_normal, zone_origin) = boundary_zone

    # Use a thin slice for the flow rate calculation
    slice = particle_spacing * 125 / 100

    # Find particles within a thin slice near the boundary face for flow rate computation
    candidates = findall(x -> dot(x - zone_origin, -face_normal) <= slice,
                         reinterpret(reshape, SVector{ndims(system), eltype(u)}, u))

    particles_in_zone = findall(particle -> boundary_zone ==
                                            current_boundary_zone(system, particle),
                                each_integrated_particle(system))

    intersect!(candidates, particles_in_zone)

    cross_sectional_area = length(candidates) * particle_spacing^(ndims(system) - 1)

    # Division inside the `sum` closure to maintain GPU compatibility
    velocity_avg = sum(candidates) do particle
        return dot(current_velocity(v, system, particle), -face_normal) / length(candidates)
    end

    # Compute volumetric flow rate: Q = v * A
    current_flow_rate = velocity_avg * cross_sectional_area

    previous_pressure = pressure[]
    previous_flow_rate = flow_rate[]
    flow_rate[] = current_flow_rate

    # Calculate new pressure according to eq. 22 in Zhang et al. (2025)
    R1 = characteristic_resistance
    R2 = peripheral_resistance
    C = compliance

    term_1 = (1 + R1 / R2) * flow_rate[]
    term_2 = C * R1 * (flow_rate[] - previous_flow_rate) / dt
    term_3 = C * previous_pressure / dt
    divisor = C / dt + 1 / R2

    pressure_new = (term_1 + term_2 + term_3) / divisor

    pressure[] = pressure_new

    return system
end

function imposed_pressure(system, pressure_model_values, boundary_zone,
                          pressure, particle)
    boundary_zone.pressure_model.is_prescribed || return pressure

    zone_id = system.boundary_zone_indices[particle]
    return pressure_model_values[zone_id].pressure[]
end

function imposed_pressure(system, pressure_model_values::Nothing, boundary_zone,
                          pressure, particle)
    return pressure
end
