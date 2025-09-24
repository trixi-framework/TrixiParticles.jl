using TrixiParticles
using Plots
using Bessels

# TODO

particle_spacing_factor = 30

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "pulsatile_channel_flow_3d.jl"),
              particle_spacing_factor=particle_spacing_factor, sol=nothing)

# Analytical velocity evolution given in eq. 18
function womersley_velocity_profile(r, t)
    omega = 1 # ω = 2π/T
    kinematic_viscosity = dynamic_viscosity / fluid_density
    alpha = pipe_radius * sqrt(omega / kinematic_viscosity)

    amp_pressure(n) = 25 #* cos(n * omega * t)

    v_x = 0.0

    for n in 1:5
        amp = im * amp_pressure(n) / (fluid_density * n * omega)

        term_1 = besselj0(alpha * sqrt(n) * im^(3 / 2) * r / pipe_radius)
        term_2 = besselj0(alpha * sqrt(n) * im^(3 / 2))

        exp_term = exp(im * n * omega * t)

        v_x += real(amp * (1 - term_1 / term_2) * exp_term)
    end

    return v_x
end

output_directory = joinpath(examples_dir(), validation_dir(), "pulsatile_channel_flow_3d")

data = TrixiParticles.CSV.read(joinpath(output_directory,
                                        "result_vx" * "_dp_$particle_spacing_factor" *
                                        ".csv"),
                               TrixiParticles.DataFrame)

times = data[!, "time"]
times_ref = [0.03, 0.05, 0.07, 0.14, 0.3, 1.0]
positions = range(-pipe_radius, pipe_radius, length=100)
data_range = 10:90
data_indices = findall(t -> t in times_ref, times)
v_x_vector = [eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]][data_indices]

# Calculate RMSEP error (eq. 17, Zhang et al.)
rmsep_run = Float64[]
for (i, t) in enumerate(times_ref)
    N = length(data_range)
    res = sum(data_range, init=0) do j
        v_x = v_x_vector[i][j]

        v_analytical = hagen_poiseuille_velocity(positions[j], t)

        # Avoid dividing by zero
        v_analytical < sqrt(eps()) && return 0.0

        rel_err = (v_analytical - v_x) / v_analytical

        return rel_err^2 / N
    end

    push!(rmsep_run, sqrt(res) * 100)
end

# RMSEP error (%) from Zhang et al. (2025)
rmsep_reference = [2.97, 1.88, 1.61, 1.5, 0.74, 0.89]

p_rmsep = scatter(collect(times_ref), rmsep_run, markersize=5, label="TrixiP")
scatter!(p_rmsep, collect(times_ref), rmsep_reference, marker=:x, markersize=5,
         markerstrokewidth=3, label="Zhang et al. (2025)")

yaxis!(p_rmsep, ylabel="RMSEP error (%)", ylims=(0, 7))
xaxis!(p_rmsep, xlabel="t", xlims=(0, 1.2))
plot!(left_margin=5Plots.mm)
plot!(right_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

@show p_rmsep

plot_range = range(-pipe_radius, pipe_radius, length=50)
v_x_plot = view(stack(v_x_vector), 1:2:100, :)
label_ = "TrixiP (" .* ["0.1" "0.3" "0.6" "0.9" "∞"] .* " s)"
line_colors = cgrad(:coolwarm, length(times_ref), categorical=true)

p = scatter(plot_range, v_x_plot, label=label_, linewidth=3, markersize=5, opacity=0.6,
            palette=line_colors.colors, legend_position=:outerright)
for t in times_ref
    label_ = t == 1.0 ? "analytical" : nothing
    plot!(p, (y) -> hagen_poiseuille_velocity(y, t), xlims=(-pipe_radius, pipe_radius),
          ylims=(-0.001, 0.005), label=label_, linewidth=3, linestyle=:dash, color=:black)
end

yaxis!(p, ylabel="x velocity (m/s)")
xaxis!(p, xlabel="y position (m)")
plot!(left_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

@show p
