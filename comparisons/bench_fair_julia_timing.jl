# Fair Rust vs Julia LMM timing: load CSV once, warmup, time MixedModels fit only.
#
#   julia comparisons/bench_fair_julia_timing.jl --data PATH --case NAME --formula "y ~ x + (1 | group)" \
#       --reml false --warmups 2 --repeats 5

using CSV
using DataFrames
using JSON
using MixedModels
using Statistics

function parse_cli_args()
    defaults = Dict(
        "data" => nothing,
        "case" => nothing,
        "formula" => nothing,
        "reml" => "false",
        "warmups" => "2",
        "repeats" => "5",
    )
    args = copy(defaults)
    raw = ARGS
    i = 1
    while i <= length(raw)
        key = raw[i]
        if startswith(key, "--")
            name = chop(key, head = 2, tail = 0)
            if !haskey(defaults, name)
                error("unknown flag --$name")
            end
            i += 1
            i > length(raw) && error("missing value for --$name")
            args[name] = raw[i]
        end
        i += 1
    end
    for required in ("data", "case", "formula")
        args[required] === nothing && error("missing required --$required")
    end
    reml = lowercase(args["reml"]) in ("true", "1", "yes")
  return (
        data = args["data"],
        case = args["case"],
        formula = args["formula"],
        reml = reml,
        warmups = parse(Int, args["warmups"]),
        repeats = parse(Int, args["repeats"]),
    )
end

function summarize(samples::Vector{Float64})
    sorted = sort(samples)
    n = length(sorted)
    median_seconds = n % 2 == 0 ? (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2 : sorted[(n + 1) ÷ 2]
    mean_seconds = mean(sorted)
    stdev_seconds = n > 1 ? std(sorted) : nothing
    return Dict(
        "min_seconds" => minimum(sorted),
        "max_seconds" => maximum(sorted),
        "mean_seconds" => mean_seconds,
        "median_seconds" => median_seconds,
        "stdev_seconds" => stdev_seconds,
    )
end

function main()
    opts = parse_cli_args()
    df = CSV.read(opts.data, DataFrame)
    for col in (:group, :plate, :sample, :batch, :cask, :Subject)
        if col in propertynames(df)
            df[!, col] = string.(df[!, col])
        end
    end
    form = eval(Meta.parse("@formula($(opts.formula))"))

    for _ in 1:opts.warmups
        fit(MixedModel, form, df, REML = opts.reml)
    end

    samples = Float64[]
    for _ in 1:opts.repeats
        elapsed = @elapsed fit(MixedModel, form, df, REML = opts.reml)
        push!(samples, elapsed)
    end

    payload = Dict(
        "implementation" => "julia",
        "case" => opts.case,
        "formula" => opts.formula,
        "reml" => opts.reml,
        "n_obs" => nrow(df),
        "warmups" => opts.warmups,
        "repeats" => opts.repeats,
        "samples_seconds" => samples,
        "summary" => summarize(samples),
    )
    println(JSON.json(payload))
end

main()
