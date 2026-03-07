using CSV
using DataFrames
using MixedModels

function main()
    file_path = joinpath("tests", "data", "dyestuff.csv")
    if !isfile(file_path)
        println("Could not find the dataset at: $file_path")
        exit(1)
    end

    println("Loading data from $file_path...")
    df = CSV.read(file_path, DataFrame)

    println("\nFitting model: Yield ~ 1 + (1 | Batch)")
    form = @formula(Yield ~ 1 + (1 | Batch))
    m1 = fit(MixedModel, form, df, REML=true)
    
    println("\n=== Model Summary ===")
    println(m1)
end

main()
