using TrixiParticles
using Plots
using Bessels

particle_spacing_factor = 30

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hagen_poiseuille_flow_3d.jl"),
              particle_spacing_factor=particle_spacing_factor, sol=nothing)

# Analytical velocity evolution given in eq. 18
function womersley_velocity_profile(r, t)
    omega = 1 # ω = 2π/T
    kinematic_viscosity = dynamic_viscosity / fluid_density
    alpha = pipe_radius * sqrt(omega / kinematic_viscosity)

    # pressure gradient magnitute for the frequency
    pressure_gradient = -25

    v_x = 0.0

    for n in 1
        amp = im * pressure_gradient / (fluid_density * n * omega)

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
                                        ".csv"), TrixiParticles.DataFrame)

times = round.(data[!, "time"], digits=2)
times_ref = round.(range(2pi, 4pi, step=0.51), digits=2)
positions = range(-pipe_radius, pipe_radius, length=100)
data_indices = findall(t -> t in times_ref, times)
v_x_vector = stack([eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]][data_indices])

velocity_grid = range(0, length(times_ref) * 5e-3, length=length(times_ref))
velocities_analytic = stack([womersley_velocity_profile.(r, times_ref) for r in positions])'

for i in eachindex(velocity_grid)
    velocities_analytic[:, i] .+= velocity_grid[i]
    v_x_vector[:, i] .+= velocity_grid[i]
end

x_analytic = [velocities_analytic[:, j] for j in 1:size(velocities_analytic, 2)]
y_analytic = fill(1:size(velocities_analytic, 1), size(velocities_analytic, 2))

x_run = [v_x_vector[1:3:100, j] for j in 1:size(v_x_vector, 2)]
y_run = fill(1:3:size(v_x_vector, 1), size(v_x_vector, 2))

xticks = (range(0, length(times_ref) * 5e-3, length=3), ["2π" "3π" "4π"])
xminorticks = collect(velocity_grid)
yticks = (range(1, 100, length=3), ["R" "0" "R"])

p = scatter(x_run, y_run, color=:red, opacity=0.5, xminorticks=6, xminorgrid=1,
            minorgridalpha=0.4, label=["numerical" "" "" "" "" "" "" "" "" "" "" "" ""])
plot!(p, x_analytic, y_analytic, legend=false, xlabel="time (s)", ylabel="radial coordinate",
      linewidth=2, linestyle=:dash, xticks=xticks, yticks=yticks, color=:black,
      size=(1000, 300), label=["analytical" "" "" "" "" "" "" "" "" "" "" "" ""])

plot!(p, left_margin=5Plots.mm, bottom_margin=5Plots.mm)
plot!(p, legend_position=:outerright)
p
