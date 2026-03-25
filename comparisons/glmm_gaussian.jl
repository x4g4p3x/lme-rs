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

    println("\nFitting: Reaction ~ 1 + Days + (1 | Subject) [REML, MixedModels.jl]")
    form = @formula(Reaction ~ 1 + Days + (1 | Subject))
    m = fit(MixedModel, form, df, REML = true)

    println("\n=== Model Summary ===")
    println(m)

    println("\n=== Predictions (population-level) ===")
    newdata = DataFrame(Days = [0.0, 1.0], Subject = ["308", "308"])
    beta = coef(m)
    X = hcat(ones(2), newdata.Days)
    println(X * beta)
end

main()
