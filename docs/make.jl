using SPNetworks
using Documenter

makedocs(;
    modules=[SPNetworks],
    authors="Denis Maua <denis.maua@gmail.com>",
    repo="https://github.com/denismaua/SPNetworks.jl/blob/{commit}{path}#L{line}",
    sitename="SPNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://denismaua.github.io/SPNetworks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/denismaua/SPNetworks.jl",
)
