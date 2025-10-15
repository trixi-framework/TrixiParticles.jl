"""
    EnergyCalculatorCallback(system, semi; interval=1,
                             eachparticle=(n_integrated_particles(system) + 1):nparticles(system),
                             only_compute_force_on_fluid=false)

Callback to integrate the work done by particles in a
[`TotalLagrangianSPHSystem`](@ref).

With the default arguments it tracks the energy contribution of the clamped particles
that follow a [`PrescribedMotion`](@ref). By selecting different particles it can also be
used to measure the work done by the structure on the surrounding fluid.

- **Prescribed/clamped motion energy** (default) -- monitor only the clamped particles by
    leaving `eachparticle` at its default range
    `(n_integrated_particles(system) + 1):nparticles(system)`.
- **Fluid load measurement** -- set `eachparticle=eachparticle(system)` together with
    `only_compute_force_on_fluid=true` to accumulate the work that the entire structure
    exerts on the surrounding fluid (useful for drag or lift estimates).

Internally the callback integrates the instantaneous power, i.e. the dot product between
the force exerted by the particle and its prescribed velocity, using an explicit Euler
time integration scheme.

The accumulated value can be retrieved via [`calculated_energy`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Arguments
- `system`: The [`TotalLagrangianSPHSystem`](@ref) whose particles should be monitored.
- `semi`: The [`Semidiscretization`](@ref) that contains `system`.

# Keywords
- `interval=1`: Interval (in number of time steps) at which to compute the energy.
                It is recommended to keep this at `1` (every time step) or small (≤ 5)
                to limit time integration errors in the energy integral.
- `eachparticle=(n_integrated_particles(system) + 1):nparticles(system)`: Iterator
                selecting which particles contribute. The default includes all clamped
                particles in the system; pass `eachparticle(system)` to include every particle.
- `only_compute_force_on_fluid=false`: When `true`, only interactions with
                fluid systems are accounted for. Combined with
                `eachparticle=eachparticle(system)`, this accumulates the work that the
                entire structure exerts on the fluid, which is useful for drag or lift
                estimates.

# Examples
```jldoctest; output = false, setup = :(system = RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend()))
# Create an energy calculator callback that is called every 2 time steps
energy_cb = EnergyCalculatorCallback(system, semi; interval=2)

# After the simulation, retrieve the calculated energy
total_energy = calculated_energy(energy_cb)

# output
0.0
```
"""
struct EnergyCalculatorCallback{T, DV, EP}
    interval                      :: Int
    t                             :: T # Time of last call
    energy                        :: T
    system_index                  :: Int
    dv                            :: DV
    eachparticle                  :: EP
    only_compute_force_on_fluid   :: Bool
end

function EnergyCalculatorCallback(system, semi; interval=1,
                                  eachparticle=(n_integrated_particles(system) + 1):nparticles(system),
                                  only_compute_force_on_fluid=false)
    ELTYPE = eltype(system)
    system_index = system_indices(system, semi)

    # Allocate buffer to write accelerations for all particles (including clamped ones)
    dv = allocate(semi.parallelization_backend, ELTYPE, (ndims(system), nparticles(system)))

    cb = EnergyCalculatorCallback(interval, Ref(zero(ELTYPE)), Ref(zero(ELTYPE)),
                                  system_index, dv, eachparticle,
                                  only_compute_force_on_fluid)

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(cb, cb, save_positions=(false, false))
end

"""
    calculated_energy(cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})

Get the current calculated energy from an [`EnergyCalculatorCallback`](@ref).

# Arguments
- `cb`: The `DiscreteCallback` returned by [`EnergyCalculatorCallback`](@ref).

# Examples
```jldoctest; output = false, setup = :(system = RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend()))
# Create an energy calculator callback
energy_cb = EnergyCalculatorCallback(system, semi)

# After the simulation, retrieve the calculated energy
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

    system = semi.systems[callback.system_index]
    update_energy_calculator!(energy, system, callback.eachparticle,
                              callback.only_compute_force_on_fluid, callback.dv,
                              v_ode, u_ode, semi, t, dt)

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return integrator
end

function update_energy_calculator!(energy, system, eachparticle,
                                   only_compute_force_on_fluid, dv,
                                   v_ode, u_ode, semi, t, dt)
    @trixi_timeit timer() "calculate energy" begin
        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" begin
            # Don't create sub-timers here to avoid cluttering the timer output
            @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)
        end

        set_zero!(dv)

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        foreach_system(semi) do neighbor_system
            if only_compute_force_on_fluid && !(neighbor_system isa AbstractFluidSystem)
                # Not a fluid system, ignore this system
                return
            end

            v_neighbor = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)

            interact!(dv, v, u, v_neighbor, u_neighbor,
                      system, neighbor_system, semi,
                      integrate_tlsph=true, # Required when using split integration
                      eachparticle=eachparticle)
        end

        if !only_compute_force_on_fluid
            @threaded semi for particle in eachparticle
                add_acceleration!(dv, particle, system)
                add_source_terms_inner!(dv, v, u, particle, system, source_terms(system), t)
            end
        end

        for particle in eachparticle
            velocity = current_velocity(v, system, particle)
            dv_particle = extract_svector(dv, system, particle)

            # The force on the clamped particle is mass times acceleration
            F_particle = system.mass[particle] * dv_particle

            # To obtain energy, we need to integrate the instantaneous power.
            # Instantaneous power is force applied BY the particle times its velocity.
            # The force applied BY the particle is the negative of the force applied ON it.
            energy[] += dot(-F_particle, velocity) * dt
        end
    end

    return energy
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})
    @nospecialize cb # reduce precompilation time

    ELTYPE = eltype(cb.affect!.energy)
    print(io, "EnergyCalculatorCallback{$ELTYPE}(interval=", cb.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})
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
