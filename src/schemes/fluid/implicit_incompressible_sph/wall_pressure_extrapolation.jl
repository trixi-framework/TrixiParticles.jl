# Coupling of IISPH with Adami and Bernoulli walls (`IISPHExtrapolatingWall`).
#
# The extrapolated wall pressure is an affine function of the fluid pressure,
#     p_b = sum_f E_bf p_f + e_b,
# with the normalized kernel weights E_bf and the hydrostatic, dynamic and offset terms e_b.
# The wall density is constant, since IISPH fluids are incompressible.
# The wall pressure is extrapolated from the current pressure iterate in every iteration
# of the pressure solve. It enters the PPE through the sums d_ij * p_j, like the pressure
# of `PressureBoundaries`. After the solve, the regular boundary update extrapolates the
# solved pressure for the force calculation.

# Whether the pressure of `system` is extrapolated in every iteration of the pressure solve
@inline iisph_wall_in_solve(system, semi) = false

@inline function iisph_wall_in_solve(wall::IISPHExtrapolatingWall, semi)
    return any(fluid -> fluid isa ImplicitIncompressibleSPHSystem &&
                        has_system_interaction(fluid, wall, semi), semi.systems)
end

@inline function has_iisph_wall_extrapolation(semi)
    return any(system -> iisph_wall_in_solve(system, semi), semi.systems)
end

function check_iisph_wall_configuration(systems, interaction_matrix)
    for (fluid_index, fluid) in pairs(systems)
        fluid isa ImplicitIncompressibleSPHSystem || continue

        for (wall_index, wall) in pairs(systems)
            wall isa IISPHExtrapolatingWall || continue

            interaction = interaction_matrix[fluid_index, wall_index]
            is_enabled_interaction(interaction) || continue

            if !isnothing(wall.boundary_model.state_equation)
                throw(ArgumentError("the boundary model of system $wall_index must not " *
                                    "have a state equation, since it interacts with the " *
                                    "incompressible `ImplicitIncompressibleSPHSystem` " *
                                    "$fluid_index. Without a state equation, the wall " *
                                    "density is the initial density."))
            end

            # The pressure solve assumes the pressure force of the default interaction
            if interaction !== true
                throw(ArgumentError("`ImplicitIncompressibleSPHSystem` $fluid_index must " *
                                    "use the default interaction with the " *
                                    "pressure-extrapolating wall system $wall_index"))
            end
        end
    end

    return nothing
end

update_iisph_wall_pressures!(mode, semi, v_ode, u_ode) = semi

function update_iisph_wall_pressures!(::WallExtrapolationIISPHSolve, semi, v_ode, u_ode)
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        iisph_wall_in_solve(system, semi) || return

        extrapolate_iisph_wall_pressure!(system, v, u, v_ode, u_ode, semi)
    end

    return semi
end

function extrapolate_iisph_wall_pressure!(wall, v, u, v_ode, u_ode, semi)
    (; boundary_model) = wall

    # The no-slip velocity does not depend on the pressure and is not recomputed here
    extrapolate_boundary_pressure!(boundary_model, wall, v, u, v_ode, u_ode, semi, nothing)

    @threaded semi for particle in eachparticle(wall)
        set_normalized_boundary_pressure!(boundary_model, particle)
    end

    return wall
end

# Use this function to avoid passing closures to Polyester.jl with `@threaded`
@inline function set_normalized_boundary_pressure!(boundary_model, particle)
    p = normalized_boundary_pressure(boundary_model, particle)
    @inbounds boundary_model.pressure[particle] = p

    return boundary_model
end

# The prediction uses the no-slip velocity of the walls in the viscous force.
# Compute it from the current fluid velocity. It is otherwise only computed after the solve.
function refresh_iisph_wall_velocities!(semi, v_ode, u_ode)
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        iisph_wall_in_solve(system, semi) || return

        refresh_wall_velocity!(system, system.boundary_model.viscosity, v, u, v_ode, u_ode,
                               semi)
    end

    return semi
end

refresh_wall_velocity!(wall, viscosity::Nothing, v, u, v_ode, u_ode, semi) = wall

function refresh_wall_velocity!(wall, viscosity, v, u, v_ode, u_ode, semi)
    (; boundary_model) = wall

    # This also accumulates the unnormalized wall pressure. The pressure solve extrapolates
    # the wall pressure again in every iteration.
    extrapolate_boundary_pressure!(boundary_model, wall, v, u, v_ode, u_ode, semi,
                                   viscosity)

    @threaded semi for particle in eachparticle(wall)
        update_extrapolated_wall_velocity!(boundary_model.cache, viscosity, wall, v,
                                           particle)
    end

    return wall
end
