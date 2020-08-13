# Implementation of SPN2MILP algorithm
import AlgebraicDecisionDiagrams
# Aliases
const ADD = AlgebraicDecisionDiagrams
const MLExpr = ADD.MultilinearExpression

"""
    spn2milp(spn::SumProductNetwork)

Translates sum-product network `spn` into MAP-equivalent mixed-integer linear program.
Require that sum nodes have exactly two children.
"""
function spn2milp(spn::SumProductNetwork, ordering::Union{Nothing,Array{<:Integer}}=nothing)
    # obtain scope of every node
    scopes = SumProductNetworks.scopes(spn)
    # Extract ADDs for each variable
    ## Colect ids of sum nodes
    sumnodes = filter(i -> issum(spn[i]), 1:length(spn))
    ## Create a bucket for each sum node / latent variable
    buckets = Dict{Int,Array{ADD.DecisionDiagram{MLExpr}}}( i => [] for i in sumnodes ) 
    # TODO: Apply min-fill or min-degree heuristic to obtain better elimination ordering
    # variable elimination sequence
    if isnothing(ordering) # if no ordering is given, obtain one
        ordering = sort(sumnodes, rev=true) # eliminate bottom-most variables first
    end
    @assert length(ordering) == length(sumnodes) 
    vorder = Dict{Int,Int}() # ordering of elimination of each variable (inverse mapping)
    for i=1:length(ordering)
        vorder[ordering[i]] = i
    end
    ## First obtain ADDs for manifest variables
    offset = 0 # offset to apply to variable indices at ADD leaves
    vdims = SumProductNetworks.vardims(spn) # var id => no. of values
    for var in sort(scopes[1])
        # Extract ADD for variable var
        α = ADD.reduce(extractADD!(Dict{Int,ADD.DecisionDiagram{MLExpr}}(), spn, 1, var, scopes, offset))
        offset += vdims[var]
        # get index of bottom-most variable (highest id of a sum node)
        # id = maximum(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))  
        # get index of lowest variable according to elimination ordering
        i,id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), Base.filter(n -> isa(n,ADD.Node), collect(α)))  
        @assert length(buckets[id]) == 0 # TODO iterate until finding an empty bucket      
        # associate ADD to corresponding bucket
        push!(buckets[id],α)
    end
    ## Then build ADDs for sum nodes (latent variables)
    for id in sumnodes
            # construct ADD
        α = ADD.Node(id,MLExpr(spn[id].weights[1]),MLExpr(spn[id].weights[2]))
        # associate ADD to corresponding bucket
        push!(buckets[id],α)
    end
    # To map each expression in a leaf into a fresh monomial
    cache = Dict{MLExpr,MLExpr}()
    function new_monomial(e::MLExpr) 
        get!(cache,e,MLExpr(1.0,offset+length(cache)+1))
    end
    # To print out constraints using the apply operation
    dummy = MLExpr(1)
    function print_constraint(e1::MLExpr, e2::MLExpr) 
        println(e1, " = ", e2)
        dummy
    end
    # Run variable elimination to generate constraints
    for i = 1:(length(ordering)-1)
        var = ordering[i] # variable to eliminate
        printstyled("Eliminate: ", var, "\n"; color = :red)
        α = reduce(*, buckets[var]; init = ADD.Terminal(MLExpr(1.0)))
        α = ADD.marginalize(α, var)        
        println(α)
        # Obtain copy with modified leaves
        β = ADD.apply(new_monomial, α)
        println(β)
        # Print out constraint
        # TODO: replace by symbolic constraint representation (interact with JUMP / Gurobi)
        ADD.apply(print_constraint, β, α)
        # For standard bucket elimination (generates tree-decomposition)
        # scope = Base.filter(n -> isa(n,ADD.Node), collect(α))
        # if isempty(scope)
        #     id = ordering[end]
        # else
        #     _, id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), scope)  
        # end   
        # push!(buckets[id], β) 
        # printstyled("-> $id\n"; color = :green)     
        # 
        # For path decomposition
        printstyled("-> $(ordering[i+1])\n"; color = :green)     
        push!(buckets[ordering[i+1]], β)
        # for α in buckets[var]
        #     println(α)
        # end
    end
    # Objective
    # TODO: Interact with gurobi / JUMP
    α = reduce(*, buckets[ordering[end]]; init = ADD.Terminal(MLExpr(1.0)))
    α = ADD.marginalize(α, ordering[end]) 
end

"""
    extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SumProductNetwork,node::Integer,var::Integer,scopes,offset)

Extract algebraic decision diagram representing the distribution of a variable `var`, using a `cache` of ADDs and `scopes`.
"""
function extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SumProductNetwork,node::Integer,var::Integer,scopes,offset)
    if haskey(cache, node) return cache[node] end
    if issum(spn[node])
        @assert length(spn[node].children) == 2
        low = extractADD!(cache,spn,spn[node].children[1],var,scopes,offset)
        high = extractADD!(cache,spn,spn[node].children[2],var,scopes,offset)
        γ = ADD.Node(Int(node),low,high)
        cache[node] = γ
        return γ
    elseif isprod(spn[node])
        for j in spn[node].children
            if var in scopes[j]
                γ = extractADD!(cache,spn,j,var,scopes,offset)
                cache[node] = γ
                return γ
            end
        end        
    elseif isa(spn[node], IndicatorFunction) # leaf
        @assert spn[node].scope == var
        stride = convert(Int, offset + spn[node].value)
        γ = ADD.Terminal(MLExpr(1.0,stride))
        cache[node] = γ
        return γ
    elseif isa(spn[node], CategoricalDistribution) # leaf
        @assert spn[node].scope == var
        expr = mapreduce(i -> MLE(spn[node].values[i], offset + i), +, 1:length(spn[node].values) )
        γ = ADD.Terminal(expr)
        cache[node] = γ
        return γ
    else
        @error "Unsupported node type: $(typeof(spn[node]))."
    end
end