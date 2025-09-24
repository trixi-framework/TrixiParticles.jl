using TrixiParticles
using Plots
using Bessels

# TODO

particle_spacing_factor = 30

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hagen_poiseuille_flow_3d.jl"),
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

# line_colors = cgrad(:coolwarm, length(time_range), categorical=true)
# p = plot(palette=line_colors.colors)
# for t in time_range
#     plot!(p, (r) -> womersley_velocity_profile(r, t), xlims=(-pipe_radius, pipe_radius),
#           ylims=(-5e-3, 5e-3), label="t = $(round(t, digits=3))", linewidth=3,
#           linestyle=:dash)
# end
# p

# Zeitpunkte und Radialpositionen
times = range(0, 2π, length=13)  # 12 Zeitpunkte über 2 Perioden
r_vals = range(-pipe_radius, pipe_radius, length=100)
velocity_grid = range(0, 13 * 5e-3, length=13)
velocities = stack([womersley_velocity_profile.(r, times) for r in r_vals])
velocities_shifted = copy(velocities')

for i in eachindex(velocity_grid)
    velocities_shifted[:, i] .+= velocity_grid[i]
end

A = velocities_shifted
xs = [A[:, j] for j in 1:size(A, 2)]
ys = fill(1:size(A, 1), size(A, 2))
xticks = (range(0, 13 * 5e-3, length=3), ["0" "pi" "2pi"])   # Positionen + Labels
yticks = (range(1, 100, length=3), ["R" "0" "R"])             # jede 20. Zeile

p = plot(xs, ys, legend=false, xlabel="time", ylabel="radial coordinate",
         xticks=xticks, yticks=yticks, color=:black, size=(900, 300))

plot!(p, left_margin=5Plots.mm)
plot!(p, bottom_margin=5Plots.mm)
p

# # Oberer Plot: Druckgradient
# p1 = plot(range(0, 4π, length=100), t -> cos(t),
#           xlabel="Time (s)", ylabel="Pressure gradient",
#           title="", linewidth=2, legend=false,
#           xlims=(0, 4π), grid=true)

# # Xticks bei Vielfachen von π
# # plot!(p1, xticks=([0, 2π, 3π, 4π], ["0", "2π", "3π", "4π"]))

# # # Unterer Plot: Geschwindigkeitsprofile
# # p2 = plot(xlabel="Velocity (m/s)", ylabel="Radial coordinate",
# #           legend=:topright, grid=true, ylims=(-pipe_radius, pipe_radius),
# #           gridwidth=1, gridcolor=:gray, gridalpha=0.3)

# # # Geschwindigkeitsprofile für verschiedene Zeitpunkte plotten
# # for (i, t) in enumerate(times)
# #     velocities = [womersley_velocity_profile(abs(r), t) for r in r_vals]

# #     if i == 1
# #         plot!(p2, velocities, r_vals, linewidth=2, color=:red,
# #               label="Analytical", alpha=0.9)
# #         scatter!(p2, velocities[1:15:end], r_vals[1:15:end],
# #                  markersize=2, color=:red, markerstrokewidth=0,
# #                  label="Numerical", alpha=0.7, markershape=:circle)
# #     else
# #         plot!(p2, velocities, r_vals, linewidth=2, color=:red,
# #               label="", alpha=0.9)
# #         scatter!(p2, velocities[1:15:end], r_vals[1:15:end],
# #                  markersize=2, color=:red, markerstrokewidth=0,
# #                  label="", alpha=0.7, markershape=:circle)
# #     end
# # end

# # # Beide Plots kombinieren
# # plot(p1, p2, layout=(2, 1), size=(800, 600),
# #      left_margin=5Plots.mm, bottom_margin=5Plots.mm)
