#!/usr/bin/env julia
using ReadVTK, Glob, Statistics, Printf, Plots

gr()

function parse_pvd(pvd_file::AbstractString)
    lines = readlines(pvd_file)
    times = Float64[]
    files = String[]
    for line in lines
        m = match(r"timestep=\"([^\"]+)\".*file=\"([^\"]+)\"", line)
        if m !== nothing
            push!(times, parse(Float64, m.captures[1]))
            push!(files, m.captures[2])
        end
    end
    return times, files
end

function read_vtu_velocity(file::AbstractString)
    vtk = ReadVTK.VTKFile(file)
    points = ReadVTK.get_points(vtk)
    coords = convert(Array{Float64, 2}, points[1:2, :])
    point_data = ReadVTK.get_point_data(vtk)
    keys_list = collect(Base.keys(point_data))
    idx = findfirst(k -> occursin("velocity", k), keys_list)
    idx === nothing && error("No velocity field found in $file")
    vel = ReadVTK.get_data(point_data[keys_list[idx]])
    vel_arr = eltype(vel) <: AbstractArray ?
              hcat([collect(ReadVTK.get_data(d)) for d in vel]...) :
              convert(Array{Float64, 2}, vel)
    return coords, vel_arr
end

function average_profile(y::Vector{Float64}, u::Vector{Float64})
    order = sortperm(y)
    y_sorted = y[order]
    u_sorted = u[order]
    unique_y = Float64[]
    mean_u = Float64[]
    i = 1
    while i <= length(y_sorted)
        yi = y_sorted[i]
        j = i
        while j <= length(y_sorted) && isapprox(y_sorted[j], yi; atol=1e-10, rtol=0.0)
            j += 1
        end
        push!(unique_y, yi)
        push!(mean_u, mean(u_sorted[i:(j - 1)]))
        i = j
    end
    return unique_y, mean_u
end

function carreau_yasuda_nu(gamma, nu0, nu_inf, lambda_cy, a_yasuda, n_cy)
    return nu_inf +
           (nu0 - nu_inf) *
           (1.0 + (lambda_cy * gamma)^a_yasuda)^((n_cy - 1.0) / a_yasuda)
end

function solve_shear_rate_from_stress(tau, rho0, nu0, nu_inf, lambda_cy, a_yasuda, n_cy)
    tau <= 0 && return 0.0
    f(gamma) = rho0 *
               carreau_yasuda_nu(gamma, nu0, nu_inf, lambda_cy, a_yasuda, n_cy) *
               gamma - tau
    lo = 0.0
    hi = 1.0
    while f(hi) < 0.0
        hi *= 2.0
        hi > 1e12 && error("Failed to bracket root for tau=$tau")
    end
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) <= 1e-12 * max(tau, 1.0)
            return mid
        elseif fmid > 0
            hi = mid
        else
            lo = mid
        end
    end
    return 0.5 * (lo + hi)
end

function analytical_ux_profile(ys, n_powerlaw, h, rho0, nu0, nu_inf, lambda_cy,
                               a_yasuda, dpdx)
    s_vals = sort(unique(abs.(ys .- 0.5 * h)))
    gamma_vals = similar(s_vals)
    for i in eachindex(s_vals)
        tau = dpdx * s_vals[i]
        gamma_vals[i] = solve_shear_rate_from_stress(tau, rho0, nu0, nu_inf,
                                                     lambda_cy, a_yasuda,
                                                     n_powerlaw)
    end
    u_of_s = zeros(length(s_vals))
    for i in (length(s_vals) - 1):-1:1
        ds = s_vals[i + 1] - s_vals[i]
        u_of_s[i] = u_of_s[i + 1] + 0.5 * (gamma_vals[i + 1] + gamma_vals[i]) * ds
    end
    ux = Vector{Float64}(undef, length(ys))
    function linear_interp_clamped(xs, ys, x)
        x <= xs[1] && return ys[1]
        x >= xs[end] && return ys[end]
        i = searchsortedlast(xs, x)
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    end
    for (k, y) in enumerate(ys)
        s = abs(y - 0.5 * h)
        ux[k] = linear_interp_clamped(s_vals, u_of_s, s)
    end
    return ux
end

function compute_errors(u_num, u_analytic)
    err = u_num .- u_analytic
    l2 = sqrt(sum(abs2, err) / length(err))
    rel_l2 = l2 / (sqrt(sum(abs2, u_analytic) / length(u_analytic)) + eps())
    maxabs = maximum(abs.(err))
    return l2, rel_l2, maxabs
end

function plot_carreau(out_root::AbstractString="out_poiseuille_carreau")
    n_dirs = sort(glob(joinpath(out_root, "n_*")))
    isempty(n_dirs) && error("No n_* directories found under $out_root")

    h = 1.0
    rho0 = 1000.0
    nu0 = 1.0e-3
    nu_inf = 0.0
    a_yasuda = 2.0
    re = 200.0
    u_ref = re * nu0 / h
    dpdx = 8.0 * rho0 * u_ref^2 / (re * h)
    lambda_cy = h / max(u_ref, eps())
    profile_plot = plot(title="Final profile",
                        xlabel="y / H",
                        ylabel="u_x / u_max",
                        legend=:bottomright,
                        xlim=(0.0, 1.0),
                        ylim=(0.0, 1.0),
                        size=(600, 600))
    mean_plot = plot(title="Domain-mean u_x vs time",
                     xlabel="time",
                     ylabel="mean(u_x)",
                     legend=:bottomright,
                     size=(600, 450))
    error_plot = plot(title="Relative L2 error vs analytical profile",
                      xlabel="time",
                      ylabel="relative L2 error",
                      legend=:bottomright,
                      size=(600, 450))

    for n_dir in n_dirs
        pvdfile = joinpath(n_dir, "fluid_1.pvd")
        if !isfile(pvdfile)
            @warn "Missing PVD file for $n_dir, skipping"
            continue
        end
        times, files = parse_pvd(pvdfile)
        errors = Vector{NTuple{3, Float64}}()
        mean_us = Float64[]
        y_norm = Float64[]
        u_norm = Float64[]
        u_analytic_sph_final = Float64[]
        n_val = parse(Float64, replace(basename(n_dir), "n_" => ""))

        for (t, file) in zip(times, files)
            coords, vel = read_vtu_velocity(joinpath(n_dir, file))
            ys = vec(coords[2, :])
            u_num = vec(vel[1, :])
            mean_u = mean(u_num)
            push!(mean_us, mean_u)

            y_unique, u_avg = average_profile(ys, u_num)
            u_analytic = analytical_ux_profile(y_unique, n_val, h, rho0, nu0,
                                               nu_inf, lambda_cy, a_yasuda, dpdx)
            l2, rel_l2, maxabs = compute_errors(u_avg, u_analytic)
            push!(errors, (l2, rel_l2, maxabs))

            if t == times[end]
                u_max = maximum(u_avg)
                y_norm = y_unique ./ h
                u_norm = u_avg ./ u_max
                u_analytic_sph_final = u_analytic ./ u_max
            end
        end

        # combined normalized final profile
        plot!(profile_plot, y_norm, u_norm;
              label="n=$(n_val)", linewidth=1, linestyle=:solid, marker=:none)

        # mean velocity time trace
        plot!(mean_plot, times, mean_us;
              label="n=$(n_val)", marker=:circle, markersize=3, linewidth=3)

        # per-n analytical comparison and error history
        y_unique = y_norm
        u_num_norm = u_norm
        u_analytic_sph_norm = u_analytic_sph_final
        error_times = times
        l2vals = [e[1] for e in errors]
        rel_l2vals = [e[2] for e in errors]
        maxvals = [e[3] for e in errors]

        per_case_fig = plot(layout=(@layout [a{0.55h}; b{0.45h}]), size=(900, 800))
        plot!(per_case_fig[1], y_unique, u_num_norm;
              label="numeric", linewidth=2, linestyle=:solid, color=:blue)
        plot!(per_case_fig[1], y_unique, u_analytic_sph_norm;
              label="analytical", linestyle=:solid, linewidth=2, color=:purple)
        xlabel!(per_case_fig[1], "y / H")
        ylabel!(per_case_fig[1], "u_x / u_max")
        title!(per_case_fig[1], "n=$(n_val) final profile")
        xlims!(per_case_fig[1], (0.0, 1.0));
        ylims!(per_case_fig[1], (0.0, 1.0))

        plot!(per_case_fig[2], error_times, rel_l2vals;
              label="relative L2", linewidth=2, color=:black)
        xlabel!(per_case_fig[2], "time")
        ylabel!(per_case_fig[2], "relative L2 error")
        title!(per_case_fig[2], "n=$(n_val) relative error vs time")

        # combined error comparison across all n values
        plot!(error_plot, error_times, rel_l2vals;
              label="n=$(n_val)", linewidth=2)

        savefig(per_case_fig, joinpath(n_dir, "carreau_comparison_and_error.png"))

        csvfile = joinpath(n_dir, "carreau_error_data.csv")
        open(csvfile, "w") do io
            println(io, "time,mean_u,L2,rel_L2,maxabs")
            for (t, m, e) in zip(times, mean_us, errors)
                @printf(io, "%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        t, m, e[1], e[2], e[3])
            end
        end
    end

    savefig(profile_plot, joinpath(out_root, "carreau_final_profiles.png"))
    savefig(mean_plot, joinpath(out_root, "carreau_mean_velocity.png"))
    savefig(error_plot, joinpath(out_root, "carreau_error_comparison.png"))
    println("Saved profile comparison to: ",
            joinpath(out_root, "carreau_final_profiles.png"))
    println("Saved mean velocity comparison to: ",
            joinpath(out_root, "carreau_mean_velocity.png"))
    println("Saved error comparison to: ",
            joinpath(out_root, "carreau_error_comparison.png"))
end

plot_carreau(length(ARGS) > 0 ? ARGS[1] : "out_poiseuille_carreau")
