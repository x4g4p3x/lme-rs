# Please install the required packages before running:
# import Pkg; Pkg.add(["CSV", "DataFrames", "MixedModels"])

using CSV
using DataFrames
using MixedModels

function main()
    file_path = joinpath("tests", "data", "sleepstudy.csv")
    if !isfile(file_path)
        println("Could not find the dataset at: $file_path")
        println("Please run this example from the root of the lme-rs repository.")
        exit(1)
    end

    println("Loading data from $file_path...")
    
    # 1. Load the dataset
    df = CSV.read(file_path, DataFrame)
    println("Successfully loaded $(nrow(df)) rows.")

    # 2. Fit the Linear Mixed Model
    println("\nFitting model: Reaction ~ 1 + Days + (1 + Days | Subject)")
    
    # MixedModels.jl uses a macro for the formula
    form = @formula(Reaction ~ 1 + Days + (1 + Days | Subject))
    
    # Fit the Linear Mixed Model (REML = true by default in fit)
    m1 = fit(MixedModel, form, df, REML=true)
    
    # 3. Print the summary
    println("\n=== Model Summary ===")
    println(m1)

    # 4. Generate Predictions
    println("\n=== Predictions ===")
    println("Generating predictions for Subject 308 at Days 0, 1, 5, and 10...")
    
    # Create new data
    newdata = DataFrame(
        Days = [0.0, 1.0, 5.0, 10.0],
        Subject = ["308", "308", "308", "308"]
    )
    
    # Predict (Population-level)
    # MixedModels.jl typically predicts exactly what is in the data, or requires the model matrix.
    # To get population-level predictions we can extract the fixed effects and compute:
    
    # \mu = X \beta
    beta = coef(m1)
    
    # Manual X matrix matching formula: Intercept, Days
    X = hcat(ones(4), newdata.Days)
    
    preds = X * beta
    
    println("Predictions (Population-level):")
    println(preds)
end

main()
