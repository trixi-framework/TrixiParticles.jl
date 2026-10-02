abstract type AbstractPressureModel end

"""
    RCRWindkesselModel(; characteristic_resistance, peripheral_resistance, compliance)

The `RCRWindkesselModel` is a biomechanical lumped-parameter representation
that captures the relationship between pressure and flow in pulsatile systems (e.g. in vascular systems)
and is used to compute the pressure in a [`BoundaryZone`](@ref).
It is derived from an electrical circuit analogy and consists of three elements:

- characteristic resistance (``R_1``): Represents the proximal resistance at the vessel entrance.
  It models the immediate pressure drop that arises at the entrance of a vessel segment,
  either due to a geometric narrowing or to a mismatch in characteristic impedance between adjacent segments.
  A larger ``R_1`` produces a sharper initial pressure rise at the onset of flow.
- peripheral resistance (``R_2``): Represents the distal resistance,
  which controls the sustained outflow into the peripheral circulation and thereby determines the level of the mean pressure.
  A high ``R_2`` maintains a higher pressure (reduced outflow), whereas a low ``R_2`` allows a faster pressure decay.
- compliance (``C``): Connected in parallel with ``R_2`` and represents the capacity of elastic walls
  to store and release volume; in other words, it models the "stretchiness" of the vessel walls.
  Analogous to a capacitor in an electrical circuit, it absorbs blood when pressure rises and releases it during diastole.
  The presence of ``C`` smooths pulsatile flow and produces a more uniform outflow profile.

Lumped-parameter models for the vascular system are well described in the literature (e.g. [Westerhof2008](@cite)).
A practical step-by-step procedure for identifying the corresponding model parameters is provided by [Gasser2021](@cite).

# Keywords
- `characteristic_resistance`: characteristic resistance (``R_1``)
- `peripheral_resistance`: peripheral resistance (``R_2``)
- `compliance`: compliance (``C``)
"""
struct RCRWindkesselModel{ELTYPE <: Real, P, FR} <: AbstractPressureModel
    characteristic_resistance :: ELTYPE
    peripheral_resistance     :: ELTYPE
    compliance                :: ELTYPE
    pressure                  :: P
    flow_rate                 :: FR
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function RCRWindkesselModel(; characteristic_resistance, peripheral_resistance, compliance)
    pressure = Ref(zero(compliance))
    flow_rate = Ref(zero(compliance))
    return RCRWindkesselModel(characteristic_resistance, peripheral_resistance, compliance,
                              pressure, flow_rate)
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

function update_pressure_model!(system, v, u, semi, dt)
    # Avoid division by zero: skip update
    dt < sqrt(eps()) && return system

    if any(pm -> isa(pm, AbstractPressureModel), system.cache.pressure_reference_values)
        @trixi_timeit timer() "update pressure model" begin
            calculate_pressure!(system, dt)
        end
    end

    return system
end

function calculate_pressure!(system, dt)
    (; pressure_reference_values, boundary_zones_flow_rate) = system.cache

    foreach_noalloc_zip(pressure_reference_values,
                        boundary_zones_flow_rate) do (pressure_model, flow_rate)
        calculate_pressure!(pressure_model, system, flow_rate[], dt)
    end

    return system
end

function calculate_pressure!(pressure_model, system, current_flow_rate, dt)
    return pressure_model
end

function calculate_pressure!(pressure_model::RCRWindkesselModel, system,
                             current_flow_rate, dt)
    (; characteristic_resistance, peripheral_resistance, compliance,
     flow_rate, pressure) = pressure_model

    previous_pressure = pressure[]
    # Flow rate still stored in `pressure_model` from the previous time step
    previous_flow_rate = flow_rate[]
    # Flow rate calculated earlier in this time step in `calculate_flow_rate!`
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

function (pressure_model::RCRWindkesselModel)(x, t)
    return pressure_model.pressure[]
end

@doc raw"""
    ImpedanceOutletPressure(; reference_velocity, impedance, reference_pressure=0.0)

Pressure model for an outlet [`BoundaryZone`](@ref) that lets waves leave the domain
without reflecting them back, while keeping the pressure level fixed.
See [Non-reflecting outlet](@ref impedance_outlet) for a step-by-step explanation.

In every time step, the pressure in the boundary zone is set to
```math
p = p_{\text{ref}} + Z \left( \bar{u} - u_{\text{ref}} \right),
```
where ``\bar{u} = Q / A`` is the mean outflow velocity, that is, the volumetric
flow rate ``Q`` out of the domain divided by the area ``A`` of the boundary face.
When the fluid leaves the domain with the expected velocity ``u_{\text{ref}}``,
the pressure is ``p_{\text{ref}}``.
When it leaves faster, the pressure is increased, which slows it down, and vice versa.
With the impedance ``Z = \rho_0 c``, where ``\rho_0`` is the reference density and
``c`` is the speed of sound of the fluid, this is exactly the relation between pressure
and velocity in a wave that moves out of the domain.
Such waves therefore pass the outlet as if the domain continued beyond it.

The flow rate is computed at the `sample_points` of the [`BoundaryZone`](@ref).
These points are automatically moved to where the boundary model applies the pressure
(see [Non-reflecting outlet](@ref impedance_outlet)).

# Keywords
- `reference_velocity`:     Expected mean outflow velocity ``u_{\text{ref}}``
                            (positive when the fluid leaves the domain).
                            When the outflow is fed by an inflow, this is the inflow rate
                            divided by the area of the outlet face.
- `impedance`:              Impedance ``Z``. Use `fluid_density * sound_speed` for an
                            outlet that does not reflect waves.
- `reference_pressure=0.0`: Pressure ``p_{\text{ref}}`` when the fluid leaves the domain
                            with the velocity `reference_velocity`.

# Examples
```jldoctest; output=false
fluid_density = 1000.0
sound_speed = 15.0

pressure_model = ImpedanceOutletPressure(; reference_velocity=1.0,
                                         impedance=fluid_density * sound_speed)

outflow = BoundaryZone(; boundary_face=([2.0, 0.0], [2.0, 1.0]), face_normal=(-1.0, 0.0),
                       particle_spacing=0.1, density=fluid_density, open_boundary_layers=4,
                       reference_pressure=pressure_model)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ BoundaryZone                                                                                     │
│ ════════════                                                                                     │
│ boundary type: ………………………………………… bidirectional_flow                                               │
│ #particles: ………………………………………………… 40                                                               │
│ width: ……………………………………………………………… 0.4                                                              │
│ cross sectional area: ……………………… 1.0                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct ImpedanceOutletPressure{ELTYPE <: Real, A, P, FR} <: AbstractPressureModel
    reference_pressure   :: ELTYPE
    reference_velocity   :: ELTYPE
    impedance            :: ELTYPE
    cross_sectional_area :: A  # Set when the `OpenBoundarySystem` is created
    pressure             :: P
    flow_rate            :: FR
end

function ImpedanceOutletPressure(; reference_velocity, impedance, reference_pressure=0.0)
    ELTYPE = typeof(float(impedance))

    return ImpedanceOutletPressure(convert(ELTYPE, reference_pressure),
                                   convert(ELTYPE, reference_velocity),
                                   convert(ELTYPE, impedance), nothing,
                                   Ref(convert(ELTYPE, reference_pressure)),
                                   Ref(zero(ELTYPE)))
end

function Base.show(io::IO, ::MIME"text/plain", pressure_model::ImpedanceOutletPressure)
    @nospecialize pressure_model # reduce precompilation time

    if get(io, :compact, false)
        show(io, pressure_model)
    else
        summary_header(io, "ImpedanceOutletPressure")
        summary_line(io, "reference_pressure", pressure_model.reference_pressure)
        summary_line(io, "reference_velocity", pressure_model.reference_velocity)
        summary_line(io, "impedance", pressure_model.impedance)
        summary_footer(io)
    end
end

function calculate_pressure!(pressure_model::ImpedanceOutletPressure, system,
                             current_flow_rate, dt)
    (; reference_pressure, reference_velocity, impedance, cross_sectional_area,
     pressure, flow_rate) = pressure_model

    flow_rate[] = current_flow_rate
    mean_velocity = current_flow_rate / cross_sectional_area
    pressure[] = reference_pressure + impedance * (mean_velocity - reference_velocity)

    return pressure_model
end

function (pressure_model::ImpedanceOutletPressure)(x, t)
    return pressure_model.pressure[]
end

# Initial pressure of the boundary zone with this pressure model
function set_initial_pressure!(pressure_model, rest_pressure)
    pressure_model.pressure[] = rest_pressure
end

function set_initial_pressure!(pressure_model::ImpedanceOutletPressure, rest_pressure)
    pressure_model.pressure[] = pressure_model.reference_pressure
end

# Called when the `OpenBoundarySystem` is created.
# Returns the boundary zone with the pressure model and the flow-rate sample points
# set up for this boundary model.
function setup_pressure_model(boundary_zone, boundary_model)
    pressure_model = boundary_zone.reference_values.reference_pressure

    return setup_pressure_model(boundary_zone, pressure_model, boundary_model)
end

setup_pressure_model(boundary_zone, pressure_model, boundary_model) = boundary_zone

function setup_pressure_model(boundary_zone, pressure_model::ImpedanceOutletPressure,
                              boundary_model)
    (; face_normal) = boundary_zone
    (; sample_points) = boundary_zone.cache

    # Without sample points, `create_cache_open_boundary` throws an error
    isnothing(sample_points) && return boundary_zone

    (; reference_pressure, reference_velocity, impedance, pressure,
     flow_rate) = pressure_model
    cross_sectional_area = boundary_zone.cache.cross_sectional_area

    # Create a new pressure model, so that the original one remains unchanged
    pressure_model_new = ImpedanceOutletPressure(reference_pressure, reference_velocity,
                                                 impedance, cross_sectional_area,
                                                 Ref(pressure[]), Ref(flow_rate[]))
    reference_values_new = (; boundary_zone.reference_values...,
                            reference_pressure=pressure_model_new)
    boundary_zone_new = @set boundary_zone.reference_values = reference_values_new

    offset = pressure_application_offset(boundary_model, boundary_zone)

    # The boundary pressure acts at the boundary face, where the sample points are
    isnothing(offset) && return boundary_zone_new

    # Move the sample points along the face normal to the plane at the distance `offset`
    # downstream of the boundary face, where the boundary model applies the pressure.
    # Note that `face_normal` points into the fluid domain and `zone_origin` lies in the
    # boundary face.
    # This must be a new array, so that the original boundary zone remains unchanged.
    sample_points_new = copy(sample_points)
    for point in axes(sample_points_new, 2)
        position = extract_svector(sample_points, Val(length(face_normal)), point)
        distance = dot(boundary_zone.zone_origin - position, face_normal)
        position_new = position - (offset - distance) * face_normal
        for dim in eachindex(position_new)
            sample_points_new[dim, point] = position_new[dim]
        end
    end

    return @set boundary_zone_new.cache.sample_points = sample_points_new
end

# Distance downstream of the boundary face at which the boundary model applies
# the boundary pressure.
# The mirroring and characteristics models set the pressure of all particles in the
# boundary zone, so the boundary pressure acts at the boundary face.
# Return `nothing` to keep the sample points at the boundary face.
pressure_application_offset(boundary_model, boundary_zone) = nothing
# See `dynamical_pressure.jl` for `BoundaryModelDynamicalPressureZhang`
