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
struct EnergyCalculatorCallback{T, DV, EP}
    interval                      :: Int
    t                             :: T # Time of last call
    energy                        :: T
    system_index                  :: Int
    dv                            :: DV
    eachparticle                  :: EP
    only_compute_force_on_fluid   :: Bool
end

function EnergyCalculatorCallback{ELTYPE}(system, semi; interval=1,
                                          eachparticle=eachparticle(system),
                                          only_compute_force_on_fluid=false) where {ELTYPE <: Real}
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
