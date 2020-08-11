# Implementation of SPN2MILP algorithm

import AlgebraicDecisionDiagrams
# Aliases
const ADD = AlgebraicDecisionDiagrams
const MLExpr = ADD.MultilinearExpression

"""
    spn2milp(spn::SumProductNetwork)

Translates sum-product network `spn` into MAP-equivalent mixed-integer linear program
"""
function spn2milp(spn::SumProductNetwork)
    # obtain scope of every node
    # TODO: move this to function scope in Core.jl
    scopes = Array{Array{Int}}(undef, length(spn))
    for i = length(spn):-1:1
        node = spn[i]
        # println(i, " : ", node)
        if isleaf(node)
            scopes[i] = Int[node.scope]
        else # can probably be done more efficiently
            scopes[i] = Base.reduce(union, map(j -> scopes[j], node.children)) 
        end
    end
    # Extract ADDs for each variable
    ## Colect ids of sum nodes
    sumnodes = filter(i -> issum(spn[i]), 1:length(spn))
    ## Create a bucket for each sum node
    buckets = [ ADD.DecisionDiagram{MLExpr}[] for _=1:length(sumnodes) ]
    ## First process ADDs for manifest variables
    for var in scopes[1] 
        # Extract ADD for variable var
        α = ADD.reduce(extractADD!(Dict{Int,ADD.DecisionDiagram{MLExpr}}(), spn, 1, var, scopes))
        # get index of bottom-most variable (highest id of a sum node)
        id = maximum(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))        
        # associate ADD to corresponding bucket
        i = findfirst(isequal(id), sumnodes)
        push!(buckets[i],α)
    end
    ## TODO Then extract ADDs for sum nodes
    for (i,id) in enumerate(sumnodes)
        # construct ADD
        α = ADD.Node(id,MLExpr(spn[id].weights[1]),MLExpr(spn[id].weights[2]))
        # associate ADD to corresponding bucket
        push!(buckets[i],α)
    end
    #TODO: 
    # - Apply min-fill or min-degree heuristic to obtain elimination ordering
    # - Run variable elimination
    # - Interact with gurobi
    buckets
end

"""
    extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SumProductNetwork,node::Integer,var::Integer,scopes)

Extract algebraic decision diagram representing the distribution of a variable `var`, using a `cache` of ADDs and `scopes`.
"""
function extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SumProductNetwork,node::Integer,var::Integer,scopes)
    if haskey(cache, node) return cache[node] end
    if issum(spn[node])
        @assert length(spn[node].children) == 2
        low = extractADD!(cache,spn,spn[node].children[1],var,scopes)
        high = extractADD!(cache,spn,spn[node].children[2],var,scopes)
        γ = ADD.Node(Int(node),low,high)
        cache[node] = γ
        return γ
    elseif isprod(spn[node])
        for j in spn[node].children
            if var in scopes[j]
                γ = extractADD!(cache,spn,j,var,scopes)
                cache[node] = γ
                return γ
            end
        end        
    else # leaf
        @assert spn[node].scope == var
        no_vars = length(scopes[1])
        stride = convert(Int,no_vars*(var-1) + spn[node].value)
        γ = ADD.Terminal(MLExpr(1.0,stride))
        cache[node] = γ
        return γ
    end
end