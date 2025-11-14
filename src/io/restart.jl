function save_config(semi, sol; output_directory="out", filename="config")

    file = joinpath(output_directory, filename*".jld2")

    t_sol = sol.t
    u_sol = sol.u

    JLD2.@save file t_sol u_sol semi

    return
end

function load_ode(tspan; input_directory="out", filename="config")
    file = joinpath(input_directory, filename*".jld2")

    JLD2.@load file t_sol u_sol semi

    t_start = tspan[1]
    t_start_index = findfirst(t -> t == t_start, t_sol)

    u_start = u_sol[t_start_index]

    semi_updated = restart_with!(semi, (u=[u_start],))

    ode = semidiscretize(semi_updated, tspan)

    return ode
end
