include("cases.jl")

project = dirname(Base.active_project())
renderer = joinpath(@__DIR__, "render.jl")

for config in CASES
    hasproperty(config, :panels) && isempty(config.panels) && continue
    run(`$(Base.julia_cmd()) --project=$project $renderer $(config.name)`)
end

run(`$(Base.julia_cmd()) --project=$project $(joinpath(@__DIR__, "make_overview.jl"))`)
