# Please install the required packages before running:
# import Pkg; Pkg.add(["CSV", "DataFrames", "MixedModels"])

using CSV
using DataFrames
using MixedModels

function main()
    file_path = joinpath("tests", "data", "pastes.csv")
    if !isfile(file_path)
        println("Could not find the dataset at: $file_path")
        println("Please run this example from the root of the lme-rs repository.")
        exit(1)
    end

    println("Loading data from $file_path...")
    
    # 1. Load the dataset
    df = CSV.read(file_path, DataFrame)

    # 2. Fit the Linear Mixed Model
    println("\nFitting model: strength ~ 1 + (1 | batch) + (1 | batch:cask)")
    
    # MixedModels.jl evaluates nested effectively as batch + batch:cask or uses zerocorr(batch&cask)
    form = @formula(strength ~ 1 + (1 | batch) + (1 | batch&cask))
    
    m1 = fit(MixedModel, form, df, REML=true)
    
    # 3. Print the summary
    println("\n=== Model Summary ===")
    println(m1)

    # 4. Generate Predictions
    println("\n=== Predictions ===")
    println("Generating predictions for population-level...")
    
    # Predict (Population-level)
    # \mu = X \beta
    beta = coef(m1)
    X = ones(3, 1) # Intercept only for fixed effects
    
    preds = X * beta
    
    println("Predictions (Population-level):")
    println(preds)
end

main()
