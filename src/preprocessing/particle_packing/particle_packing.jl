include("signed_distance.jl")
include("system.jl")
include("rhs.jl")

function start_particle_packing(ic, shape; smoothing_kernel, smoothing_length,
                                background_pressure, tlsph, time_integrator, dtmax, tspan,
                                info_callback, solution_saving_callback, maxiters,
                                neighborhood_search)
    packing_system = ParticlePackingSystem(ic, smoothing_kernel, smoothing_length;
                                           boundary=shape, background_pressure, tlsph)

    semi = Semidiscretization(packing_system)

    if neighborhood_search
        nhs_particles = get_neighborhood_search(packing_system, semi)
        nhs_faces = FaceNeighborhoodSearch(nhs_particles)
    else
        nhs_faces = TrivialNeighborhoodSearch{ndims(packing_system)}(1.0, eachface(shape))
    end

    packing_system.nhs_faces = nhs_faces

    ode = semidiscretize(semi, tspan)

    callbacks = CallbackSet(UpdateCallback(), solution_saving_callback, info_callback)
    sol = solve(ode, time_integrator;
                save_everystep=false, maxiters, callback=callbacks, dtmax)

    v_ode, u_ode = sol.u[end].x

    u = wrap_u(u_ode, packing_system, semi)

    # TODO: Remove particles that are closer than `max_distance` to any particle
    max_distance = 0.25ic.particle_spacing
    too_close = find_too_close_particles(u, max_distance)

    if isempty(too_close)
        ic.coordinates .= u

        return ic
    end
    not_too_close = setdiff(eachparticle(packing_system), too_close)

    coordinates = u[:, not_too_close]
    velocity = ic.velocity[:, not_too_close]
    mass = ic.mass[not_too_close]
    density = ic.density[not_too_close]
    pressure = ic.pressure[not_too_close]

    @info "$(length(too_close)) removed particles that are too close together"

    return InitialCondition{ndims(ic)}(coordinates, velocity, mass, density, pressure,
                                       ic.particle_spacing)
end
