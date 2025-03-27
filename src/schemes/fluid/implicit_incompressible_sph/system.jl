
struct ImplicitIncompressibleSPHSystem{NDIMS, ELTYPE <: Real, IC, MA, P, K,
                                   V, PF, SRFN, Dens, PV, D, A, SD, S} <:
       FluidSystem{NDIMS, IC}
    initial_condition                 :: IC 
    mass                              :: MA     # Array{ELTYPE, 1}
    pressure                          :: P      # Array{ELTYPE, 1} 
    smoothing_kernel                  :: K 
    smoothing_length                  :: ELTYPE 
    acceleration                      :: SVector{NDIMS, ELTYPE} 
    viscosity                         :: V 
    pressure_acceleration_formulation :: PF
    transport_velocity                :: Nothing # TODO
    surface_normal_method             :: SRFN
    buffer                            :: Nothing
    density                           :: Dens
    v_adv                             :: PV  
    d                                 :: D
    a                                 :: A
    sum_dj                            :: SD
    s_term                            :: S
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function ImplicitIncompressibleSPHSystem(initial_condition,
                                     smoothing_kernel, smoothing_length;
                                     pressure_acceleration=nothing,
                                     buffer_size=nothing,
                                     viscosity=nothing,
                                     acceleration=ntuple(_ -> 0.0,
                                                         ndims(smoothing_kernel)),
                                     reference_particle_spacing=0.0)
    buffer = isnothing(buffer_size) ? nothing :
             SystemBuffer(nparticles(initial_condition), buffer_size)

    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)
    n_particles = nparticles(initial_condition)

    mass = copy(initial_condition.mass)
    pressure = copy(initial_condition.pressure)

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    # Make acceleration an SVector
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    pressure_acceleration = pressure_acceleration_summation_density


    density = copy(initial_condition.density)
    a = zeros(ELTYPE, n_particles)
    d = zeros(ELTYPE, NDIMS, n_particles)
    v_adv = zeros(ELTYPE, NDIMS, n_particles)
    sum_dj = zeros(ELTYPE, NDIMS, n_particles)
    s_term = zeros(ELTYPE, n_particles)
    return ImplicitIncompressibleSPHSystem(initial_condition, mass, pressure,
                                       smoothing_kernel, smoothing_length,
                                       acceleration_, viscosity,
                                       pressure_acceleration, nothing, nothing,
                                       nothing, density, v_adv, d, a, sum_dj, s_term)
end

# New NoS: density zum cache hinzufügen, damit die density im system gespeichert wird
function create_cache_density(initial_condition)
    density = similar(initial_condition.density)

    return (; density)
end

function reset_callback_flag!(system::ImplicitIncompressibleSPHSystem)
    return system
end

function Base.show(io::IO, system::ImplicitIncompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "ImplicitIncompressibleSPHSystem{", ndims(system), "}(")
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.source_terms)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::ImplicitIncompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "ImplicitIncompressibleSPHSystem{$(ndims(system))}")
        if system.buffer isa SystemBuffer
            summary_line(io, "#particles", nparticles(system))
            summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        else
            summary_line(io, "#particles", nparticles(system))
        end
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@propagate_inbounds function particle_pressure(v, system::ImplicitIncompressibleSPHSystem,
                                               particle)
    return system.pressure[particle]
end

@propagate_inbounds function predicted_velocity(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.v_adv, system, particle)
end

@propagate_inbounds function predicted_velocity(system::BoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end


@propagate_inbounds function get_d(system::ImplicitIncompressibleSPHSystem, d, particle)
    return extract_svector(d, system, particle)
end

@propagate_inbounds function get_d(system::BoundarySystem, d, particle)
    return extract_svector(d, system, particle)
end

@propagate_inbounds function get_sum_dj(system::ImplicitIncompressibleSPHSystem, sum_dj, particle)
    return extract_svector(sum_dj, system, particle)
end

@propagate_inbounds function get_sum_dj(system::BoundarySystem, sum_dj, particle)
    return extract_svector(sum_dj, system, particle)
end

#New NoS: New density function for IISPH
@propagate_inbounds function particle_density(v, system::ImplicitIncompressibleSPHSystem, particle)
    return system.density[particle]
end

#TODO: Als SVector zurückgeben?
function calculate_sum_dj(system :: ImplicitIncompressibleSPHSystem, particle, density, time_step, grad_kernel)
    m_b = hydrodynamic_mass(system, particle)
    p_b = system.pressure[particle]
    rho_b = density[particle]
    return SVector(-time_step^2 * m_b / rho_b^2 * p_b * grad_kernel)
end

function calculate_sum_dj(system :: BoundarySystem, particle, density, time_step, grad_kernel)
    return zero(SVector{ndims(system), eltype(system)})
end

#TODO
function calculate_sterm(system, neighbor_system:: ImplicitIncompressibleSPHSystem, particle, neighbor, density, pressure, sum_dj, d, grad_kernel_ab, grad_kernel_ba, time_step)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    sum_da = get_sum_dj(system, sum_dj, particle)
    d_b = get_d(neighbor_system, d, neighbor)
    p_a = pressure[particle]
    p_b = pressure[neighbor]
    sum_db = get_sum_dj(neighbor_system, sum_dj, neighbor)
    dba = -time_step^2 * hydrodynamic_mass(system, particle) / density[particle]^2 * grad_kernel_ba
    return m_b * dot(sum_da - d_b * p_b - (sum_db - dba * p_a), grad_kernel_ab)
end

#TODO
function calculate_sterm(system, neighbor_system:: BoundarySystem, particle, neighbor, density, pressure, sum_dj, d, grad_kernel_ab, grad_kernel_ba, time_step)
    sum_da = get_sum_dj(system, sum_dj, particle)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    return m_b * dot(sum_da, grad_kernel_ab)
end

#TODO: Als SVector zurückgeben?
function calculate_dji(system, ::ImplicitIncompressibleSPHSystem, particle, density, grad_kernel, time_step)
    return SVector(-time_step^2 * hydrodynamic_mass(system, particle) / density[particle]^2 * grad_kernel)
end

function calculate_dji(system, ::BoundarySystem, particle, density, grad_kernel, time_step)
    return zero(SVector{ndims(system), eltype(system)})
end

#TODO: Was machen mit dem Soundspeed?
@inline system_sound_speed(system::ImplicitIncompressibleSPHSystem) = 1000

#TODO: Calculate pressure values 
function update_quantities!(system::ImplicitIncompressibleSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    
    #New NoS: density muss irgendwo gespeichert werden. Für die Einheitlichkeit habe ich im cache wieder density hinzugefügt
    # und greife dann auf diese density zu in der summation_density Funktion. 

    # feste time step size
    time_step = 0.00001
    
    # get necessary fields
    density = system.density # density
    d = system.d # array for the d_ii values
    set_zero!(d) # set to zero in each function call 
    a = system.a # array for the a_ii values
    set_zero!(a) # set to zero in each function call 
    v_adv = system.v_adv # predicted velocity


    # ziehe die aktuellen Geschwindigkeiten
    v_particle_system = wrap_v(v_ode, system, semi)

    set_zero!(density)
    summation_density!(system, semi,  u, u_ode, density)
    
    v_adv .= v_particle_system # set to current velocity
    @trixi_timeit timer() "first loop" foreach_system(semi) do neighbor_system
        # Get neighbor system u and v values 
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        # get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
        # get neighborhood_search
        nhs = get_neighborhood_search(system, neighbor_system, semi)
        
        foreach_point_neighbor(system, neighbor_system,
                                system_coords, neighbor_system_coords,
                                nhs;
                                points=each_moving_particle(system)) do particle,
                                                                        neighbor,
                                                                        pos_diff,
                                                                        distance

            # each moving particle sind alle particle die integriert werden, was bei einem Fluid alle sind
            # CALC VISCOSITY:

            #TODO: Was tun mit dem sound speed???
            sound_speed = system_sound_speed(system)

            m_a = hydrodynamic_mass(system, particle)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)


            #New Density function for IISPH in density_calculator.jl
            rho_a = @inbounds particle_density(v_particle_system, system, particle)
            rho_b = @inbounds particle_density(v_neighbor_system, neighbor_system, neighbor)
 

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)


            dv_viscosity_ = @inbounds dv_viscosity(system, neighbor_system,
                                v_particle_system, v_neighbor_system,
                                particle, neighbor, pos_diff, distance,
                                sound_speed, m_a, m_b, rho_a, rho_b,
                                grad_kernel)


            # CALC d_ii (fluid and boundary particles are considered)
            for i in ndims(system)
                d[i, particle] -= time_step^2 * m_b / rho_a^2 * grad_kernel[i]
            end

            # CALC f_adv (external forces) and v_adv
            for i in ndims(system)
                v_adv[i, particle] += time_step * (dv_viscosity_[i] + system.acceleration[i])
            end
            #Alternative: add_acceleration!(F_adv, particle, system)
        end    
    end
    # get pressure values
    pressure = system.pressure

    # Calculation of the a_ii-values 
    @trixi_timeit timer() "second loop" foreach_system(semi) do neighbor_system
        # Get neighbor system u and v values 
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        # get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
        # get neighborhood_search
        nhs = get_neighborhood_search(system, neighbor_system, semi)
        
        foreach_point_neighbor(system, neighbor_system,
                                system_coords, neighbor_system_coords,
                                nhs;
                                points=each_moving_particle(system)) do particle,
                                                                        neighbor,
                                                                        pos_diff,
                                                                        distance
            #Set pressure p0 to a half of the previous/current pressure value
            pressure[particle] = 0.5 * pressure[particle]

            grad_kernel = smoothing_kernel_grad(system, -pos_diff, distance)
            # compute dji for the calculation of aii (TODO: Nur dji's Berechen für FLuid Particle j, nicht für Boundary Particles)
            dji = calculate_dji(system, neighbor_system, particle, density, grad_kernel, time_step)
                               
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            # compute a_ii
            a[particle] += m_b * dot((get_d(system, d, particle) - dji), smoothing_kernel_grad(system, pos_diff, distance))
        end
    end

    foreach_system(semi) do neighbor_system
        # Get neighbor system u and v values 
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        # get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
        # get neighborhood_search
        nhs = get_neighborhood_search(system, neighbor_system, semi)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs,
                                points=each_moving_particle(system)) do particle, neighbor,
                                                                        pos_diff, distance
            
            # calculate predicted density
            v_adv_diff = predicted_velocity(system, particle) - predicted_velocity(neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)

            density[particle] += time_step * m_b * dot(v_adv_diff, grad_kernel)
        end 
    end
    #ITERATION
    sum_dj = system.sum_dj
    w = 0.5
    rest_density = 1000.0
    s_term = system.s_term
    sum_dj .= 0.0
    s_term .= 0.0
    l = 0
    while l < 2
        @trixi_timeit timer() "while loop 1" foreach_system(semi) do neighbor_system
            # Get neighbor system u and v values 
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
            # get coordinates
            system_coords = current_coordinates(u, system)
            neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
            # get neighborhood_search
            nhs = get_neighborhood_search(system, neighbor_system, semi)

            foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs;
            points=each_moving_particle(system)) do particle,
                                                    neighbor,
                                                    pos_diff,
                                                    distance

                grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)
                # gives only a value for fluid particles (0 for boundary particles) 
                sum_dj_ = calculate_sum_dj(neighbor_system, neighbor, density, time_step, grad_kernel)
                for i in 1:ndims(system)
                    sum_dj[i, particle] += sum_dj_[i] 
                end
            end
        end

        @trixi_timeit timer() "while loop 2" foreach_system(semi) do neighbor_system
            # Get neighbor system u and v values 
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
            # get coordinates
            system_coords = current_coordinates(u, system)
            neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
            # get neighborhood_search
            nhs = get_neighborhood_search(system, neighbor_system, semi)
            @trixi_timeit timer() "calc s_term loop" foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs;
                                    points=each_moving_particle(system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance
                
                grad_kernel_ab = smoothing_kernel_grad(system, pos_diff, distance)
                grad_kernel_ba = smoothing_kernel_grad(system, -pos_diff, distance)
                s_term[particle] += calculate_sterm(system, neighbor_system, particle, neighbor, density, pressure, sum_dj, d, grad_kernel_ab, grad_kernel_ba, time_step)                      
            end
        end
        for particle in eachparticle(system)
            #pressure[particle] = (1-w) * pressure[particle] + w * 1/a[particle] * (rest_density - density[particle] - s_term[particle])
            #version with pressure clamping (no negative pressure values)
            pressure[particle] = max((1-w) * pressure[particle] + w * 1/a[particle] * (rest_density - density[particle] - s_term[particle]), 0)
            #pressure[particle] = 0.0
            #density[particle] = 1000.0
        end
        l += 1
    end
    #=
    #print(density)
    println("Density Minimum: ", minimum(density))
    println("Density Maximum: ", maximum(density))
    println("Pressure Minimum: ", minimum(pressure))
    println("Pressure Maximum: ", maximum(pressure))
    #println(a)
    #println(size(a))
    println("a Minimum: ", minimum(a))
    println("a Maximum: ", maximum(a))
    #println(system.s_term)
    #println(size(system.s_term))
    println("s_term Minimum: ", minimum(system.s_term))
    println("s_term Maximum: ", maximum(system.s_term))
    println("------------------------------------------------------")
    =#
    return system
end


function write_v0!(v0, system::ImplicitIncompressibleSPHSystem)
    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(system.initial_condition.velocity)
    copyto!(v0, indices, system.initial_condition.velocity, indices)
    return v0
end
