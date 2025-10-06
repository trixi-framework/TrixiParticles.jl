"""
    EnergyCalculatorCallback{ELTYPE}(; interval=1)

Callback to calculate the energy contribution from clamped particles in
[`TotalLagrangianSPHSystem`](@ref)s.

This callback computes the work done by clamped particles that are moving according to a
[`PrescribedMotion`](@ref).
The energy is calculated by integrating the instantaneous power, which is the dot product
of the force exerted by the clamped particle and its prescribed velocity.
The callback uses a simple explicit Euler time integration for the energy calculation.

The accumulated energy can be retrieved using [`calculated_energy`](@ref).

!!! info "TLSPH System Configuration"
    The [`TotalLagrangianSPHSystem`](@ref) must be created with
    `use_with_energy_calculator_callback=true` to prepare the system for energy
    calculation by allocating the necessary cache for clamped particles.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Arguments
- `ELTYPE`: The floating point type used for time and energy calculations.
            This should match the floating point type used in the
            [`TotalLagrangianSPHSystem`](@ref), which can be obtained via `eltype(system)`.

# Keywords
- `interval=1`: Interval (in number of time steps) at which to compute the energy.
                It is recommended to set this either to `1` (every time step) or to a small
                integer (e.g., `2` or `5`) to reduce time integration errors
                in the energy calculation.

# Examples
```jldoctest; output = false, setup = :(structure = RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); smoothing_kernel = WendlandC2Kernel{2}(); smoothing_length = 1.0; young_modulus = 1e6; poisson_ratio = 0.3; n_clamped_particles = 0; clamped_particles_motion = nothing)
# Create TLSPH system with `use_with_energy_calculator_callback=true`
system = TotalLagrangianSPHSystem(structure, smoothing_kernel, smoothing_length,
                                  young_modulus, poisson_ratio,
                                  n_clamped_particles=n_clamped_particles,
                                  clamped_particles_motion=clamped_particles_motion,
                                  use_with_energy_calculator_callback=true) # Important!

# Create an energy calculator that runs every 2 time steps
energy_cb = EnergyCalculatorCallback{eltype(system)}(; interval=2)

# After the simulation, retrieve the calculated energy
total_energy = calculated_energy(energy_cb)

# output
0.0
```
"""
struct EnergyCalculatorCallback{T}
    interval :: Int
    t        :: T # Time of last call
    energy   :: T
end

function EnergyCalculatorCallback{ELTYPE}(; interval=1) where {ELTYPE <: Real}
    cb = EnergyCalculatorCallback(interval, Ref(zero(ELTYPE)), Ref(zero(ELTYPE)))

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(cb, cb, save_positions=(false, false))
end

"""
    calculated_energy(cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})

Get the current calculated energy from an [`EnergyCalculatorCallback`](@ref).

# Arguments
- `cb`: The `DiscreteCallback` returned by [`EnergyCalculatorCallback`](@ref).

# Examples
```jldoctest; output = false
# Before the simulation: create the callback
energy_cb = EnergyCalculatorCallback{Float64}()

# After the simulation: get the calculated energy
total_energy = calculated_energy(energy_cb)

# output
0.0
```
"""
function calculated_energy(cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})
    return cb.affect!.energy[]
end

# `condition`
function (callback::EnergyCalculatorCallback)(u, t, integrator)
    (; interval) = callback

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (callback::EnergyCalculatorCallback)(integrator)
    # Determine time step size as difference to last time this callback was called
    t = integrator.t
    dt = t - callback.t[]

    # Update time of last call
    callback.t[] = t

    semi = integrator.p
    v_ode, u_ode = integrator.u.x
    energy = callback.energy

    foreach_system(semi) do system
        update_energy_calculator!(energy, v_ode, u_ode, system, semi, t, dt)
    end

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return integrator
end

function update_energy_calculator!(energy, v_ode, u_ode, system, semi, t, dt)
    return energy
end

function update_energy_calculator!(energy, v_ode, u_ode,
                                   system::AbstractStructureSystem, semi, t, dt)
    @trixi_timeit timer() "calculate energy" begin
        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" begin
            # Don't create sub-timers here to avoid cluttering the timer output
            @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)
        end

        dv_clamped = system.cache.dv_clamped
        set_zero!(dv_clamped)

        # Create a view of `dv_clamped` that can be indexed as
        # `(n_integrated_particles(system) + 1):nparticles(system)`.
        # We pass this to `interact!` below so that it can compute
        # the forces on the clamped particles without having to allocate
        # a large matrix for all particles.
        dv_offset = OffsetMatrix(dv_clamped, n_integrated_particles(system))
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        eachparticle = (n_integrated_particles(system) + 1):nparticles(system)

        foreach_system(semi) do neighbor_system
            v_neighbor = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)

            interact!(dv_offset, v, u, v_neighbor, u_neighbor,
                      system, neighbor_system, semi,
                      integrate_tlsph=true, # Required when using split integration
                      eachparticle=eachparticle)
        end

        @threaded semi for particle in eachparticle
            add_acceleration!(dv_offset, particle, system)
            add_source_terms_inner!(dv_offset, v, u, particle, system,
                                    source_terms(system), t)
        end

        for particle in (n_integrated_particles(system) + 1):nparticles(system)
            velocity = current_velocity(nothing, system, particle)
            dv_particle = extract_svector(dv_offset, system, particle)

            # The force on the clamped particle is mass times acceleration
            F_particle = system.mass[particle] * dv_particle

            # To obtain energy, we need to integrate the instantaneous power.
            # Instantaneous power is force done BY the particle times prescribed velocity.
            # The work done BY the particle is the negative of the work done ON it.
            energy[] -= dot(F_particle, velocity) * dt
        end
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})
    @nospecialize cb # reduce precompilation time

    ELTYPE = eltype(cb.affect!.energy)
    print(io, "EnergyCalculatorCallback{$ELTYPE}(interval=", cb.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, EnergyCalculatorCallback{T}}) where {T}
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_cb = cb.affect!
        ELTYPE = eltype(update_cb.energy)
        setup = [
            "interval" => update_cb.interval
        ]
        summary_box(io, "EnergyCalculatorCallback{$ELTYPE}", setup)
    end
end

# Data type that represents a matrix with an offset in the second dimension.
# For example, `om = OffsetMatrix(A, offset)` starts at `om[:, offset + 1]`, which is
# the same as `A[:, 1]` and it ends at `om[:, offset + size(A, 2)]`, which is
# the same as `A[:, end]`.
# This is used above to compute the acceleration only for clamped particles (which are the
# last particles in the system) without having to pass a large matrix for all particles.
struct OffsetMatrix{T, M} <: AbstractMatrix{T}
    matrix::M
    offset::Int

    function OffsetMatrix(matrix, offset)
        new{eltype(matrix), typeof(matrix)}(matrix, offset)
    end
end

@inline function Base.size(om::OffsetMatrix)
    return (size(om.matrix, 1), size(om.matrix, 2) + om.offset)
end

@inline function Base.getindex(om::OffsetMatrix, i, j)
    @boundscheck checkbounds(om, i, j)
    @boundscheck j > om.offset

    return @inbounds om.matrix[i, j - om.offset]
end

@inline function Base.setindex!(om::OffsetMatrix, value, i, j)
    @boundscheck checkbounds(om, i, j)
    @boundscheck j > om.offset

    return @inbounds om.matrix[i, j - om.offset] = value
end
