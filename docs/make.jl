using SumProductNetworks
using Documenter

makedocs(;
    modules=[SumProductNetworks],
    authors="Denis Maua <denis.maua@gmail.com>",
    repo="https://github.com/denismaua/SumProductNetworks.jl/blob/{commit}{path}#L{line}",
    sitename="SumProductNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://denismaua.github.io/SumProductNetworks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/denismaua/SumProductNetworks.jl",
)
