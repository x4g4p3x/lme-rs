# Prior weights: same construction as Rust `comparisons/lmer_weighted.rs`.
# MixedModels uses the `wts` keyword for case weights (see MixedModels.jl docs).

using CSV
using DataFrames
using MixedModels

function main()
    file_path = joinpath("tests", "data", "sleepstudy.csv")
    if !isfile(file_path)
        println("Could not find the dataset at: $file_path")
        println("Run this script from the repository root.")
        exit(1)
    end

    df = CSV.read(file_path, DataFrame)
    n = nrow(df)
    wts = [0.5 + Float64((i - 1) % 5) * 0.1 for i in 1:n]

    println("\nFitting weighted model: Reaction ~ 1 + Days + (1 + Days | Subject)")
    println("Prior weights w_i = 0.5 + (row_index mod 5) * 0.1")

    form = @formula(Reaction ~ 1 + Days + (1 + Days | Subject))
    m = fit(MixedModel, form, df; REML = true, wts = wts)

    println("\n=== Model Summary ===")
    println(m)

    println("\n=== Predictions (population-level, manual fixed part) ===")
    newdata = DataFrame(Days = [0.0, 1.0, 5.0, 10.0], Subject = ["308", "308", "308", "308"])
    beta = coef(m)
    X = hcat(ones(4), newdata.Days)
    preds = X * beta
    println(preds)
end

main()
