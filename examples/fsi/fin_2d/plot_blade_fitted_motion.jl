using JSON
using Plots

input_file = joinpath(@__DIR__, "blade_motion_2mm.json")

frequency = 1.06
period = 1 / frequency
period_start = 1.0
period_end = period_start + period
ramp_duration = 0.5
spectral_harmonics = 4

data = JSON.parsefile(input_file, allownan=true)["blade_motion_structure_1"]
time = Float64.(data["time"])
values = data["values"]

blade_translation = Float64.(reduce(hcat, first.(values)))
blade_rotation = Float64.(last.(values))

period_indices = findall(t -> period_start <= t <= period_end, time)
period_time = time[period_indices]
period_translation = blade_translation[:, period_indices]
period_rotation = blade_rotation[period_indices]

function spectral_matrix(t, harmonics)
    phase = 2pi .* (t .- period_start) ./ period
    return reduce(hcat,
                  index == 0 ? ones(length(t)) :
                  isodd(index) ? cos.(((index + 1) ÷ 2) .* phase) :
                  sin.((index ÷ 2) .* phase)
                  for index in 0:(2 * harmonics))
end

fit_matrix = spectral_matrix(period_time, spectral_harmonics)
translation_coefficients = fit_matrix \ transpose(period_translation)
rotation_coefficients = fit_matrix \ period_rotation

basis_description = "[1, cos(θ), sin(θ), ..., cos($(spectral_harmonics)θ), sin($(spectral_harmonics)θ)], θ = 2π(t - $period_start) / $period"
coefficient_output = """
# Basis: $basis_description
fin_motion_period_start = $(repr(period_start))
fin_translation_x_coefficients = $(repr(Tuple(translation_coefficients[:, 1])))
fin_translation_y_coefficients = $(repr(Tuple(translation_coefficients[:, 2])))
fin_rotation_coefficients = $(repr(Tuple(rotation_coefficients)))
"""
println(coefficient_output)

translation_residual = fit_matrix * translation_coefficients -
                       transpose(period_translation)
rotation_residual = fit_matrix * rotation_coefficients - period_rotation
translation_rmse = Tuple(sqrt.(sum(abs2, translation_residual; dims=1) ./
                             length(period_indices)))
rotation_rmse = sqrt(sum(abs2, rotation_residual) / length(period_indices))
@info "Motion fit over one movement period" period_start period_end spectral_harmonics translation_rmse rotation_rmse

fit_time = range(first(time), last(time); length=2000)
evaluation_matrix = spectral_matrix(fit_time, spectral_harmonics)
fitted_translation = transpose(evaluation_matrix * translation_coefficients)
fitted_rotation = evaluation_matrix * rotation_coefficients

function smooth_ramp(t, duration)
    phase = clamp(t / duration, 0, 1)
    return phase^3 * (10 + phase * (-15 + 6phase))
end

ramp = smooth_ramp.(fit_time, ramp_duration)
fitted_translation .*= transpose(ramp)
fitted_rotation .*= ramp

translation_plot = plot(time, eachrow(blade_translation);
                        xlabel="Time (s)", ylabel="Translation (m)",
                        label=["measured x" "measured y"], color=[1 2], linewidth=2,
                        title="spectral fit of blade translation")
plot!(translation_plot, fit_time, eachrow(fitted_translation);
      label=["fitted x" "fitted y"], color=[1 2], linestyle=:dash,
      linewidth=2)

rotation_plot = plot(time, blade_rotation;
                     xlabel="Time (s)", ylabel="Rotation (rad)",
                     label="measured", color=3, linewidth=2,
                     title="spectral fit of blade rotation")
plot!(rotation_plot, fit_time, fitted_rotation;
      label="fitted", color=3, linestyle=:dash, linewidth=2)

figure = plot(translation_plot, rotation_plot;
              layout=(2, 1), size=(800, 1000), dpi=400, framestyle=:box)
