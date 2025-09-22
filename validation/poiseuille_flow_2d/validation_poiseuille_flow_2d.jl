using TrixiParticles

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              output_directory=joinpath(validation_dir(), "poiseuille_flow_2d"),
              saving_callback=nothing, tspan=(0.0, 2.0))

output_directory = joinpath(examples_dir(), validation_dir(), "poiseuille_flow_2d")

data = TrixiParticles.CSV.read(joinpath(output_directory, "result_vx.csv"),
                               TrixiParticles.DataFrame)

times = data[!, "time"]
v_x_arrays = [eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]]

rmsep_run = Float64[]
for t_target in (0.1, 0.3, 0.6, 0.9, 2.0)
    idx = argmin(abs.(times .- t_target))  # Nearest available time point

    range_ = 10:90
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
