struct RCRWindkesselModel{ELTYPE <: Real, PP, PFR, FR}
    characteristic_resistance :: ELTYPE
    peripheral_resistance     :: ELTYPE
    compliance                :: ELTYPE
    previous_pressure         :: PP
    previous_flow_rate        :: PFR
    flow_rate                 :: FR
    is_prescribed             :: Bool
end

function RCRWindkesselModel(; characteristic_resistance, peripheral_resistance, compliance,
                            is_prescribed=true)
    previous_pressure = Ref(zero(characteristic_resistance))
    previous_flow_rate = Ref(zero(characteristic_resistance))
    flow_rate = Ref(zero(characteristic_resistance))

    return RCRWindkesselModel(characteristic_resistance, peripheral_resistance, compliance,
                              previous_pressure, previous_flow_rate, flow_rate,
                              is_prescribed)
end

function update_pressure_model!(system::OpenBoundarySystem, v, u, semi, dt)
    system.has_pressure_model || return system

    calculate_flow_rate!(system, v, u)

    @threaded semi for particle in each_integrated_particle(system)
        boundary_zone = current_boundary_zone(system, particle)
        update_pressure_model!(boundary_zone.pressure_model, system,
                               system.boundary_model, boundary_zone, particle, v, u, dt)
    end
end

function update_pressure_model!(pressure_model, system, boundary_model, boundary_zone,
                                particle, v, u, dt)
    return pressure_model
end

function update_pressure_model!(pressure_model::RCRWindkesselModel,
                                system::OpenBoundarySystem,
                                boundary_model::BoundaryModelDynamicalPressureZhang,
                                boundary_zone, particle, v, u, dt)
    boundary_zone.pressure_model.is_prescribed || return pressure_model
    dt < sqrt(eps()) && return pressure_model

    (; characteristic_resistance, peripheral_resistance, compliance, flow_rate,
     previous_flow_rate, previous_pressure) = pressure_model

    term_1 = (1 + characteristic_resistance / peripheral_resistance) * flow_rate[]
    term_2 = compliance * peripheral_resistance * (flow_rate[] - previous_flow_rate[]) / dt
    term_3 = compliance * previous_pressure[] / dt
    divisor = compliance / dt + 1 / characteristic_resistance

    pressure = (term_1 + term_2 + term_3) / divisor

    set_modeled_pressure!(v, system, particle, pressure, boundary_zone, boundary_model)

    previous_pressure[] = pressure

    return pressure_model
end

function calculate_flow_rate!(system, v, u)
    for boundary_zone in system.boundary_zones
        if boundary_zone.pressure_model.is_prescribed
            calculate_flow_rate!(system, boundary_zone, v, u)
        end
    end

    return system
end

function calculate_flow_rate!(system, boundary_zone, v, u)
    (; face_normal, zone_origin, pressure_model) = boundary_zone

    # Use kernel support radius as thickness for the flow rate calculation slice
    dvolume = compact_support(system.fluid_system, system.fluid_system)

    # Find particles within a thin slice near the boundary face for flow rate computation
    candidates = findall(x -> dot(x - zone_origin, -face_normal) <= dvolume,
                         reinterpret(reshape, SVector{ndims(system), eltype(u)}, u))

    # Division inside the `sum` closure to maintain GPU compatibility
    velocity_avg = sum(candidates) do particle
        return dot(current_velocity(v, system, particle), face_normal) / length(candidates)
    end

    # Calculate total volume of particles in the slice
    volume_total = sum(candidates) do particle
        return hydrodynamic_mass(system, particle) / current_density(v, system, particle)
    end

    # Compute volumetric flow rate: Q = A * velocity_avg, where A = volume_total / dvolume
    volume_flow = velocity_avg * volume_total / dvolume

    pressure_model.previous_flow_rate[] = pressure_model.flow_rate[]
    pressure_model.flow_rate[] = volume_flow

    return system
end

# TODO: EDAC?
function set_modeled_pressure!(v, system, particle, pressure, boundary_zone,
                               boundary_model::BoundaryModelDynamicalPressureZhang)
    boundary_zone.rest_pressure[] = pressure

    return v
end
