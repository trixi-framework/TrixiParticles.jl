# Keep the CPU CSR buffers out of the fluid system, whose size affects neighbor-loop codegen.
function create_iisph_pressure_cache(systems, backend)
    backend isa KernelAbstractions.GPU && return nothing
    count(system -> system isa ImplicitIncompressibleSPHSystem, systems) == 1 ||
        return nothing

    system = systems[findfirst(system -> system isa ImplicitIncompressibleSPHSystem,
                               systems)]
    n_particles = nparticles(system)
    ELTYPE = eltype(system)
    NDIMS = ndims(system)

    walls_in_solve = filter(systems) do neighbor
        return is_iisph_pressure_boundary_system(neighbor) ||
               neighbor isa IISPHExtrapolatingWall
    end
    walls = map(walls_in_solve) do wall_system
        n_wall_particles = nparticles(wall_system)
        return (system=wall_system,
                fluid_offsets=zeros(Int, n_particles + 1),
                fluid_cursor=zeros(Int, n_particles),
                fluid_neighbor=Int[],
                fluid_d_ij=ELTYPE[],
                wall_offsets=zeros(Int, n_wall_particles + 1),
                wall_cursor=zeros(Int, n_wall_particles),
                wall_neighbor=Int[],
                wall_coef_pj=ELTYPE[],
                wall_grad_mass=ELTYPE[],
                wall_pi_coef=ELTYPE[])
    end

    return (; inv_a_ii=zeros(ELTYPE, n_particles),
            pressure_neighbor_count=zeros(Int, n_particles),
            pressure_neighbor_offsets=zeros(Int, n_particles + 1),
            pressure_neighbor_cursor=zeros(Int, n_particles),
            pressure_neighbor=Int[],
            pressure_d_ij=ELTYPE[],
            pressure_grad_mass=ELTYPE[],
            pressure_d_ji_dot_grad=ELTYPE[],
            pressure_boundary_grad_mass_sum=zeros(ELTYPE, NDIMS, n_particles), walls)
end

function create_iisph_pressure_state()
    return (last_iterations=Ref(0), total_iterations=Ref(0), max_iterations=Ref(0),
            solve_count=Ref(0), last_solve_time=Ref(0.0), total_solve_time=Ref(0.0),
            max_solve_time=Ref(0.0))
end

function update_iisph_inverse_diagonal!(system, semi)
    supports_cached_iisph_pressure_loop(semi) || return system
    (; inv_a_ii) = semi.iisph_pressure_cache
    (; a_ii) = system
    @threaded semi for particle in eachparticle(system)
        diagonal = @inbounds a_ii[particle]
        @inbounds inv_a_ii[particle] = abs(diagonal) > convert(typeof(diagonal), 1.0e-9) ?
                                       inv(diagonal) : zero(diagonal)
    end
    return system
end

function record_iisph_pressure_iterations!(semi, iterations, solve_time)
    state = semi.iisph_pressure_state
    state.last_iterations[] = iterations
    state.total_iterations[] += iterations
    state.max_iterations[] = max(state.max_iterations[], iterations)
    state.solve_count[] += 1
    state.last_solve_time[] = solve_time
    state.total_solve_time[] += solve_time
    state.max_solve_time[] = max(state.max_solve_time[], solve_time)
    return semi
end

"""
    reset_iisph_pressure_iteration_stats!(semi)

Reset the pressure solver counters. See [`iisph_pressure_iteration_stats`](@ref).
"""
function reset_iisph_pressure_iteration_stats!(semi)
    for counter in semi.iisph_pressure_state
        counter[] = zero(counter[])
    end
    return semi
end

"""
    iisph_pressure_iteration_stats(semi)

Return the last, total, maximum and average Jacobi iteration counts and solver wall
times in seconds, along with the solve count. Wall times exclude coefficient assembly.
Use [`reset_iisph_pressure_iteration_stats!`](@ref) to reset the counters.
"""
function iisph_pressure_iteration_stats(semi)
    state = semi.iisph_pressure_state
    solve_count = state.solve_count[]
    total_iterations = state.total_iterations[]
    total_solve_time = state.total_solve_time[]
    return (last_iterations=state.last_iterations[], total_iterations,
            max_iterations=state.max_iterations[], solve_count,
            average_iterations=solve_count == 0 ? 0.0 : total_iterations / solve_count,
            last_solve_time=state.last_solve_time[], total_solve_time,
            max_solve_time=state.max_solve_time[],
            average_solve_time=solve_count == 0 ? 0.0 : total_solve_time / solve_count)
end
