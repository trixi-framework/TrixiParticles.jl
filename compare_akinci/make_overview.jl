using CairoMakie
using FileIO

include("cases.jl")

const BACKGROUND = RGBf(0.94, 0.96, 0.97)
const PANEL_BACKGROUND = RGBf(0.035, 0.045, 0.065)
const TEXT_COLOR = RGBf(0.08, 0.11, 0.14)

function draw_cell!(position, cell)
    axis = Axis(position; aspect=DataAspect(), backgroundcolor=PANEL_BACKGROUND,
                title=cell.label, titlecolor=TEXT_COLOR, titlesize=17)
    hidedecorations!(axis)
    hidespines!(axis)

    if hasproperty(cell, :file)
        path = joinpath(@__DIR__, cell.file)
        isfile(path) || error("missing rendered panel: $path")
        image!(axis, rotr90(FileIO.load(path)))
    else
        text!(axis, 0.5, 0.5; text=cell.placeholder, align=(:center, :center),
              color=:white, fontsize=22, space=:relative)
        limits!(axis, 0, 1, 0, 1)
    end
    return axis
end

function make_plate(plate)
    rows, columns = plate.layout
    figure = Figure(; size=(420 * columns, 355 * rows + 130),
                    backgroundcolor=BACKGROUND, fontsize=18)
    Label(figure[1, 1:columns], plate.title; color=TEXT_COLOR, fontsize=28,
          font=:bold, tellwidth=false)

    for (index, cell) in enumerate(plate.cells)
        row = div(index - 1, columns) + 2
        column = mod1(index, columns)
        draw_cell!(figure[row, column], cell)
    end

    caption_row = rows + 2
    Label(figure[caption_row, 1:columns], plate.caption; color=TEXT_COLOR,
          fontsize=17, justification=:left, halign=:left, tellwidth=false)
    rowgap!(figure.layout, 10)
    colgap!(figure.layout, 6)

    output = joinpath(@__DIR__, plate.output)
    save(output, figure)
    println("Wrote $(plate.title) to $output")
    return output
end

function make_overview(plates)
    columns = 2
    rows = cld(length(plates), columns)
    figure = Figure(; size=(1800, 560 * rows), backgroundcolor=BACKGROUND,
                    fontsize=18)

    for (index, plate) in enumerate(plates)
        row = div(index - 1, columns) + 1
        column = mod1(index, columns)
        axis = Axis(figure[row, column]; aspect=DataAspect(),
                    backgroundcolor=BACKGROUND)
        image!(axis, rotr90(FileIO.load(joinpath(@__DIR__, plate.output))))
        hidedecorations!(axis)
        hidespines!(axis)
    end

    rowgap!(figure.layout, 12)
    colgap!(figure.layout, 12)
    output = joinpath(@__DIR__, "akinci_comparison.png")
    save(output, figure)
    println("Wrote Akinci comparison overview to $output")
    return output
end

foreach(make_plate, PLATES)
make_overview(PLATES)
