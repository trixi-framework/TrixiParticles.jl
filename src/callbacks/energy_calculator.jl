struct EnergyCalculatorCallback{T}
    interval :: Int
    t        :: T
    energy   :: T
end

function EnergyCalculatorCallback{ELTYPE}(; interval=1) where {ELTYPE <: Real}
    cb = EnergyCalculatorCallback(interval, Ref(zero(ELTYPE)), Ref(zero(ELTYPE)))

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(cb, cb, save_positions=(false, false))
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

    update_energy_calculator!(energy, v_ode, u_ode, semi.systems[1], semi, t, dt)

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return integrator
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

        dv = zero(wrap_v(v_ode, system, semi))
        dv_combined = CombinedMatrix(dv, dv_clamped)
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        eachparticle = (n_integrated_particles(system) + 1):nparticles(system)

        foreach_system(semi) do neighbor_system
            v_neighbor = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)

            interact!(dv_combined, v, u, v_neighbor, u_neighbor,
                      system, neighbor_system, semi,
                      eachparticle=eachparticle)
        end

        @threaded semi for particle in eachparticle
            # Dispatch by system type to exclude boundary systems
            add_acceleration!(dv_combined, particle, system)
            add_source_terms_inner!(dv_combined, v, u, particle, system,
                                    source_terms(system), t)
        end

        n_clamped_particles = nparticles(system) - n_integrated_particles(system)
        for clamped_particle in 1:n_clamped_particles
            particle = clamped_particle + n_integrated_particles(system)
            velocity = current_velocity(nothing, system, particle)
            dv_particle = extract_svector(dv_clamped, system, clamped_particle)

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

# Data type that combines two matrices to behave like a single matrix.
# This is used above to combine the `dv` for integrated particles and the `dv_clamped`
# for clamped particles into a single matrix that can be passed to `interact!`.
struct CombinedMatrix{T, M1, M2} <: AbstractMatrix{T}
    matrix1::M1
    matrix2::M2

    function CombinedMatrix(matrix1, matrix2)
        @assert size(matrix1, 1) == size(matrix2, 1)
        @assert eltype(matrix1) == eltype(matrix2)

        new{eltype(matrix1), typeof(matrix1), typeof(matrix2)}(matrix1, matrix2)
    end
end

@inline function Base.size(cm::CombinedMatrix)
    return (size(cm.matrix1, 1), size(cm.matrix1, 2) + size(cm.matrix2, 2))
end

@inline function Base.getindex(cm::CombinedMatrix, i, j)
    @boundscheck checkbounds(cm, i, j)

    length1 = size(cm.matrix1, 2)
    if j <= length1
        return @inbounds cm.matrix1[i, j]
    else
        return @inbounds cm.matrix2[i, j - length1]
    end
end

@inline function Base.setindex!(cm::CombinedMatrix, value, i, j)
    @boundscheck checkbounds(cm, i, j)

    length1 = size(cm.matrix1, 2)
    if j <= length1
        return @inbounds cm.matrix1[i, j] = value
    else
        return @inbounds cm.matrix2[i, j - length1] = value
    end
end
