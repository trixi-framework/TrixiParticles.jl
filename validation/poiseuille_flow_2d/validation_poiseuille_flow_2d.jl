using TrixiParticles

v_x_interpolated(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function v_x_interpolated(system::TrixiParticles.AbstractFluidSystem{2},
                          dv_ode, du_ode, v_ode, u_ode, semi, t)
    start_point = [flow_length / 2, 0.0]
    end_point = [flow_length / 2, wall_distance]

    values = interpolate_line(start_point, end_point, 100, semi, system, v_ode, u_ode;
                              cut_off_bnd=true, clip_negative_pressure=false)

    return values.velocity[1, :]
end

output_directory = joinpath(validation_dir(), "poiseuille_flow_2d")
pp_callback = PostprocessCallback(; dt=0.02,
                                  output_directory=output_directory,
                                  v_x=v_x_interpolated, filename="result_vx",
                                  write_csv=true, write_file_interval=1)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              saving_callback=nothing, tspan=(0.0, 2.0), extra_callback=pp_callback)

# Analytical velocity evolution given in eq. 16
function poiseuille_velocity(y, t)

    # Base profile (stationary part)
    base_profile = (pressure_drop / (2 * dynamic_viscosity * flow_length)) * y *
                   (y - wall_distance)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in 0:10  # Limit to 10 terms for convergence
        coefficient = (4 * pressure_drop * wall_distance^2) /
                      (dynamic_viscosity * flow_length * pi^3 * (2 * n + 1)^3)

        sine_term = sin(pi * y * (2 * n + 1) / wall_distance)

        exp_term = exp(-((2 * n + 1)^2 * pi^2 * dynamic_viscosity * t) /
                       (fluid_density * wall_distance^2))

        transient_sum += coefficient * sine_term * exp_term
    end

    # Total velocity
    v_x = base_profile + transient_sum

    return v_x
end

data = TrixiParticles.CSV.read(joinpath(output_directory, "result_vx.csv"),
                               TrixiParticles.DataFrame)

times = data[!, "time"]
v_x_arrays = [eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]]

rmsep_run = Float64[]
for t_target in (0.1, 0.3, 0.6, 0.9, 2.0)
    idx = argmin(abs.(times .- t_target))  # Nearest available time point

    range_ = 1:100
    N = length(range_)
    positions = range(0, wall_distance, length=100)
    res = sum(range_, init=0) do i
        v_x = v_x_arrays[idx][i]

        v_analytical = -poiseuille_velocity(positions[i], t_target)

        v_analytical < sqrt(eps()) && return 0.0

        rel_err = (v_analytical - v_x) / v_analytical

        return rel_err^2 / N
    end

    push!(rmsep_run, sqrt(res) * 100)
end

# RMSEP error (%) from Zhang et al. (2025)
rmsep_reference = [
    1.81,
    0.95,
    0.67,
    0.86,
    1.22
]

deviation_from_reference = rmsep_run .- rmsep_reference
