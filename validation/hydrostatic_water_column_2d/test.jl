using CairoMakie, Colors

function to_color(c)
    return isa(c, Symbol) ? parse(Colorant, string(c)) : c
end

function lighten(c, amount::Real)
  c = to_color(c)
  c_hsl = convert(HSL, c)
  new_l = clamp(c_hsl.l + amount, 0, 1)
  return HSL(c_hsl.h, c_hsl.s, new_l)
end

function darken(c, amount::Real)
  c = to_color(c)
  c_hsl = convert(HSL, c)
  new_l = clamp(c_hsl.l - amount, 0, 1)
  return HSL(c_hsl.h, c_hsl.s, new_l)
end

fig = Figure()
ax = Axis(fig[1, 1])

# Use a smaller lighten factor so that the color remains red.
l1 = lines!(ax, [0.0, 1.0], [0.0, 0.0]; color = lighten(:red, 0.15), linestyle = :solid, linewidth = 2)
l2 = lines!(ax, [0.0, 1.0], [0.0, 0.0]; color = darken(:red, 0.15),  linestyle = :dot,   linewidth = 4)

leg = Legend(fig, [l1, l2], ["Light Red", "Dark Red"], "Test Legend")
fig[2, 1] = leg

fig

save("hydrostatic_water_column_validation.svg", fig)
