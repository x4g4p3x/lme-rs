#!/usr/bin/env julia
# Format or check Julia comparison scripts (invoked from lme_ci.py).
using JuliaFormatter

function formatted_contents(file::AbstractString)
    original = read(file, String)
    dir = dirname(abspath(file))
    temp = joinpath(dir, ".fmtcheck_" * basename(file))
    try
        write(temp, original)
        format_file(temp)
        read(temp, String)
    finally
        rm(temp, force=true)
    end
end

function format_one(file::AbstractString, check::Bool)
    original = read(file, String)
    formatted = formatted_contents(file)
    if original == formatted
        return 0
    end
    if check
        println(stderr, "would reformat: $file")
        return 1
    end
    format_file(abspath(file))
    0
end

function main()
    check = "--check" in ARGS
    files = filter(!startswith("--"), ARGS)
    if isempty(files)
        exit(0)
    end

    exit_code = 0
    for file in files
        if !isfile(file)
            println(stderr, "missing file: $file")
            exit_code = 1
            continue
        end
        try
            exit_code = max(exit_code, format_one(file, check))
        catch err
            println(stderr, "failed to format $file: $err")
            exit_code = 1
        end
    end
    exit(exit_code)
end

main()
