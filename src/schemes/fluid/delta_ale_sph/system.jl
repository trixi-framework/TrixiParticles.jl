@doc raw"""
    DeltaALESPHSystem(initial_condition; smoothing_kernel, smoothing_length,
                      sound_speed, reference_density, maximum_velocity,
                      delta=0.1, acceleration=ntuple(_ -> 0.0, NDIMS),
                      viscosity=nothing, pressure_acceleration=nothing,
                      correction=nothing, source_terms=nothing)

The ``\delta``-ALE-SPH formulation by [Antuono et al. (2021)](@cite Antuono2021).

In addition to density and velocity, this system evolves the particle masses. Particle
positions are advected with the physical velocity plus the bounded shifting velocity from
equations (17)-(18) of the paper. The density and mass diffusion terms use the same
coefficient `delta`, as proposed in equation (14).

# Arguments
- `initial_condition`: [`InitialCondition`](@ref) representing the fluid particles.

# Keywords
- `sound_speed`: Artificial speed of sound ``c_0``.
- `reference_density`: Reference density ``\rho_0``.
- `maximum_velocity`: Expected maximum fluid velocity ``U_\mathrm{max}`` used by the
                      particle shifting technique.
- `smoothing_kernel`: Smoothing kernel used by the system.
- `smoothing_length`: Smoothing length used by the system.
- `delta`: Dimensionless density and mass diffusion coefficient (default: `0.1`).
- `acceleration`: Constant acceleration vector (default: zero vector).
- `viscosity`: Viscosity model (default: no viscosity).
- `pressure_acceleration`: Pressure acceleration formulation. By default, the formulation
                           for [`ContinuityDensity`](@ref) is selected.
- `correction`: Correction method (default: no correction).
- `source_terms`: Additional acceleration source terms.
"""
struct DeltaALESPHSystem{NDIMS, ELTYPE <: Real, IC, M, P, DC, SE, K, V, DD, COR,
                         PF, SC, ST, B, PR, C} <: AbstractFluidSystem{NDIMS}
    initial_condition                 :: IC
    mass                              :: M
    pressure                          :: P
    density_calculator                :: DC
    state_equation                    :: SE
    smoothing_kernel                  :: K
    acceleration                      :: SVector{NDIMS, ELTYPE}
    viscosity                         :: V
    density_diffusion                 :: DD
    correction                        :: COR
    pressure_acceleration_formulation :: PF
    shifting_technique                :: SC
    source_terms                      :: ST
    surface_tension                   :: Nothing
    surface_normal_method             :: Nothing
    buffer                            :: B
    particle_refinement               :: PR
    cache                             :: C
end

function DeltaALESPHSystem(initial_condition; smoothing_kernel, smoothing_length,
                           sound_speed, reference_density, maximum_velocity,
                           delta=0.1,
                           acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                               ndims(smoothing_kernel)),
                           viscosity=nothing, pressure_acceleration=nothing,
                           correction=nothing, source_terms=nothing)
    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)
    n_particles = nparticles(initial_condition)

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    if maximum_velocity <= 0
        throw(ArgumentError("`maximum_velocity` must be positive"))
    end

    if delta <= 0
        throw(ArgumentError("`delta` must be positive"))
    end

    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    if correction isa ShepardKernelCorrection
        throw(ArgumentError("`ShepardKernelCorrection` cannot be used with `DeltaALESPHSystem`"))
    end

    density_calculator = ContinuityDensity()
    state_equation = StateEquationCole(; sound_speed, reference_density, exponent=1)
    density_diffusion = DensityDiffusionAntuono(; delta)
    shifting_technique = DeltaALEShifting(; maximum_velocity)

    pressure_acceleration = choose_pressure_acceleration_formulation(pressure_acceleration,
                                                                     density_calculator,
                                                                     NDIMS, ELTYPE,
                                                                     correction)

    mass = copy(initial_condition.mass)
    pressure = similar(initial_condition.pressure)

    cache = (;
             create_cache_correction(correction, initial_condition.density, NDIMS,
                                     n_particles)...,
             create_cache_density_diffusion(initial_condition, density_diffusion)...,
             create_cache_shifting(initial_condition, shifting_technique)...,
             create_cache_refinement(initial_condition, nothing, smoothing_length)...,
             color=1)

    return DeltaALESPHSystem(initial_condition, mass, pressure, density_calculator,
                             state_equation, smoothing_kernel, acceleration_, viscosity,
                             density_diffusion, correction, pressure_acceleration,
                             shifting_technique, source_terms, nothing, nothing, nothing,
                             nothing, cache)
end

function Base.show(io::IO, system::DeltaALESPHSystem)
    @nospecialize system

    print(io, "DeltaALESPHSystem{", ndims(system), "}(")
    print(io, system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.correction)
    print(io, ", delta=", system.density_diffusion.delta)
    print(io, ", maximum_velocity=", system.shifting_technique.maximum_velocity)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::DeltaALESPHSystem)
    @nospecialize system

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "DeltaALESPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "smoothing length", initial_smoothing_length(system))
        summary_line(io, "sound speed", system_sound_speed(system))
        summary_line(io, "reference density", system.state_equation.reference_density)
        summary_line(io, "maximum velocity", system.shifting_technique.maximum_velocity)
        summary_line(io, "delta", system.density_diffusion.delta)
        summary_line(io, "viscosity", system.viscosity)
        summary_line(io, "correction method", system.correction |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline Base.eltype(::DeltaALESPHSystem{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

@inline v_nvariables(system::DeltaALESPHSystem) = ndims(system) + 2
@inline buffer(system::DeltaALESPHSystem) = system.buffer
@inline system_correction(system::DeltaALESPHSystem) = system.correction
@inline system_sound_speed(system::DeltaALESPHSystem) = sound_speed(system.state_equation)
@inline shifting_technique(system::DeltaALESPHSystem) = system.shifting_technique
@inline density_diffusion(system::DeltaALESPHSystem) = system.density_diffusion

@inline function current_velocity(v, system::DeltaALESPHSystem)
    return view(v, 1:ndims(system), :)
end

@inline function current_density(v, system::DeltaALESPHSystem)
    return view(v, ndims(system) + 1, :)
end

@inline function current_mass(v, system::DeltaALESPHSystem)
    return view(v, ndims(system) + 2, :)
end

@propagate_inbounds function current_mass(v, system::DeltaALESPHSystem, particle)
    return current_mass(v, system)[particle]
end

@inline current_pressure(v, system::DeltaALESPHSystem) = system.pressure

function update_quantities!(system::DeltaALESPHSystem, v, u, v_ode, u_ode, semi, t)
    copyto!(system.mass, current_mass(v, system))
    update!(system.density_diffusion, v, u, system, semi)

    return system
end

function update_pressure!(system::DeltaALESPHSystem, v, u, v_ode, u_ode, semi, t)
    compute_pressure!(system, v, semi)
    compute_correction_values!(system, system.correction, u, v_ode, u_ode, semi)
    compute_gradient_correction_matrix!(system.correction, system, u, v_ode, u_ode, semi)

    return system
end

function update_final!(system::DeltaALESPHSystem, v, u, v_ode, u_ode, semi, t; kwargs...)
    update_shifting!(system, system.shifting_technique, v, u, v_ode, u_ode, semi)
end

function compute_gradient_correction_matrix!(correction,
                                             system::DeltaALESPHSystem, u,
                                             v_ode, u_ode, semi)
    return system
end

function compute_gradient_correction_matrix!(corr::Union{GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection},
                                             system::DeltaALESPHSystem, u,
                                             v_ode, u_ode, semi)
    (; correction_matrix) = system.cache

    compute_gradient_correction_matrix!(correction_matrix, system,
                                        current_coordinates(u, system),
                                        v_ode, u_ode, semi, corr,
                                        system.smoothing_kernel)
end

@inline function apply_state_equation!(system::DeltaALESPHSystem, density, particle)
    system.pressure[particle] = system.state_equation(density)
end

function write_v0!(v0, system::DeltaALESPHSystem, ::ContinuityDensity)
    v0[ndims(system) + 1, :] = system.initial_condition.density
    v0[ndims(system) + 2, :] = system.initial_condition.mass

    return v0
end

function restart_with!(system::DeltaALESPHSystem, v, u)
    for particle in each_integrated_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
        system.initial_condition.density[particle] = v[ndims(system) + 1, particle]
        system.initial_condition.mass[particle] = v[ndims(system) + 2, particle]
        system.mass[particle] = v[ndims(system) + 2, particle]
    end

    return system
end

function sort_system!(system::DeltaALESPHSystem, v, u, perm, buffer::Nothing)
    system_coords = current_coordinates(u, system)
    system_velocity = current_velocity(v, system)
    system_density = current_density(v, system)
    system_mass = current_mass(v, system)
    system_pressure = current_pressure(v, system)

    system_coords .= system_coords[:, perm]
    system_velocity .= system_velocity[:, perm]
    system_density .= system_density[perm]
    system_mass .= system_mass[perm]
    system_pressure .= system_pressure[perm]
    system.mass .= system_mass

    return system
end

function system_data(system::DeltaALESPHSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    return (; coordinates=current_coordinates(u, system),
            velocity=current_velocity(v, system),
            mass=current_mass(v, system),
            density=current_density(v, system),
            pressure=current_pressure(v, system),
            acceleration=current_velocity(dv, system))
end

function total_mass(system::DeltaALESPHSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return sum(current_mass(wrap_v(v_ode, system, semi), system))
end

function kinetic_energy(system::DeltaALESPHSystem,
                        dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)

    return sum(each_active_particle(system)) do particle
        velocity = current_velocity(v, system, particle)
        current_mass(v, system, particle) * dot(velocity, velocity) / 2
    end
end
