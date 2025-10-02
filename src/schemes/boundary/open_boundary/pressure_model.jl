struct RCRWindkesselModel{ELTYPE <: Real}
    characteristic_resistance :: ELTYPE
    peripheral_resistance     :: ELTYPE
    compliance                :: ELTYPE
    is_prescribed             :: Bool
end

function RCRWindkesselModel(; characteristic_resistance, peripheral_resistance, compliance)

    # Compliance is analogous to a capacitor in an electric circuit.
    # In biomechanics, this models the "stretchiness" of vessels.
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

    cross_sectional_area = length(candidates) * particle_spacing^(ndims(system) - 1)

    # Division inside the `sum` closure to maintain GPU compatibility
    velocity_avg = sum(candidates) do particle
        return dot(current_velocity(v, system, particle), face_normal) / length(candidates)
    end

    # Compute volumetric flow rate: Q = v * A
    current_flow_rate = velocity_avg * cross_sectional_area

    previous_pressure = pressure[]
    previous_flow_rate = flow_rate[]
    flow_rate[] = current_flow_rate

    # Calculate new pressure according to eq. 22 in Zhang et al. (2025)
    R1 = peripheral_resistance
    R2 = characteristic_resistance
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
                          rest_pressure, particle)
    boundary_zone.pressure_model.is_prescribed || return rest_pressure

    zone_id = system.boundary_zone_indices[particle]
    return pressure_model_values[zone_id].pressure[]
end

function imposed_pressure(system, pressure_model_values::Nothing, boundary_zone,
                          pressure, particle)
    return pressure
end
