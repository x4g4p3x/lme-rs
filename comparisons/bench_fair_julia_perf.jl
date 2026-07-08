# Fair-harness Julia fit breakdown: wall time + MixedModels.jl optsum (feval, optimizer, …).
#
#   julia comparisons/bench_fair_julia_perf.jl --data PATH --case NAME \
#       --formula "y ~ x + (1 | plate) + (1 | sample)" --reml false --warmups 1

using CSV
using DataFrames
using JSON
using MixedModels

function parse_cli_args()
    defaults = Dict(
        "data" => nothing,
        "case" => nothing,
        "formula" => nothing,
        "reml" => "false",
        "warmups" => "1",
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
    )
end

function optsum_payload(os)
    feval = os.feval
    return Dict(
        "optimizer_feval" => feval,
        "optimizer_fmin" => os.fmin,
        "optimizer" => string(os.optimizer),
        "backend" => string(os.backend),
        "returnvalue" => string(os.returnvalue),
        "ftol_rel" => os.ftol_rel,
        "ftol_abs" => os.ftol_abs,
        "maxfeval" => os.maxfeval,
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

    elapsed = @elapsed m = fit(MixedModel, form, df, REML = opts.reml)
    os = m.optsum
    feval = os.feval
    mean_feval_seconds = feval > 0 ? elapsed / feval : nothing

    payload = Dict(
        "implementation" => "julia",
        "case" => opts.case,
        "formula" => opts.formula,
        "reml" => opts.reml,
        "n_obs" => nrow(df),
        "warmups" => opts.warmups,
        "fit_wall_seconds" => elapsed,
        "mean_feval_seconds" => mean_feval_seconds,
        "optsum" => optsum_payload(os),
    )
    println(JSON.json(payload))
end

main()
