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
function spn2milp(io, spn::SumProductNetwork, ordering::Union{Nothing,Array{<:Integer}}=nothing)    
    # obtain scope of every node
    scopes = scopes(spn)
    # Extract ADDs for each variable
    ## Colect ids of sum nodes
    sumnodes = filter(i -> issum(spn[i]), 1:length(spn))
    ## Create a bucket for each sum node / latent variable
    buckets = Dict{Int,Array{ADD.DecisionDiagram{MLExpr}}}( i => [] for i in sumnodes ) 
    # variable elimination sequence
    if isnothing(ordering) # if no ordering is given, obtain one
        ## Domain graph
        graph = Dict{Int,Set{Int}}( i => Set{Int}() for i in sumnodes )
    end
    ## First obtain ADDs for manifest variables
    offset = 0 # offset to apply to variable indices at ADD leaves
    vdims = vardims(spn) # var id => no. of values
    pool = ADD.DecisionDiagram{MLExpr}[]
    for var in sort(scopes[1])
        # Extract ADD for variable var
        α = ADD.reduce(extractADD!(Dict{Int,ADD.DecisionDiagram{MLExpr}}(), spn, 1, var, scopes, offset))
        offset += vdims[var] # update start index for next variable
        # Create corresponding optimization variables for leaves (interacts with Gurobi)
        map(t -> begin
            println(io, "binary ", ADD.value(t))
            end, 
                Base.filter(n -> isa(n,ADD.Terminal), collect(α))
            )          
        push!(pool, α)
        if isnothing(ordering)
            # update domain graph (connect variables in the ADD's scope)
            sc = map(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))
            for i in sc
                union!(graph[i],sc)
            end
        end        
    end
    ## Then build ADDs for sum nodes (latent variables)
    for id in sumnodes
        # construct ADD
        α = ADD.Node(id,MLExpr(spn[id].weights[1]),MLExpr(spn[id].weights[2]))
        # associate ADD to corresponding bucket
        push!(buckets[id],α)
    end
    # Find variable elimination sequence
    if isnothing(ordering)
        ordering = Int[]
        sizehint!(ordering, length(sumnodes))
        while !isempty(graph)
            # find minimum degree node -- break ties by depth/variable id
            deg, k = minimum( p -> (length(p[2]), -p[1]), graph )
            j = -k
            push!(ordering, j)
            # remove j and incident edges
            for k in graph[j]
                setdiff!(graph[k], j)
            end
            delete!(graph, j)
        end 
    end    
    @assert length(ordering) == length(sumnodes) 
    vorder = Dict{Int,Int}() # ordering of elimination of each variable (inverse mapping)
    for i=1:length(ordering)
        vorder[ordering[i]] = i
    end      
    # Add ADDs to appropriate buckets
    for α in pool
        # get index of first variable to be eliminated
        i, id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), Base.filter(n -> isa(n,ADD.Node), collect(α)))
        push!(buckets[id], α)
    end
    # release pool of ADDs to be collected by garbage collector
    pool = nothing    
    # To map each expression in a leaf into a fresh monomial
    cache = Dict{MLExpr,MLExpr}()
    bilinterms = Dict{ADD.Monomial,Int}()
    function renameleaves(e::MLExpr) 
        # get!(cache,e,MLExpr(1.0,offset+length(cache)+1))
        # If cached, do nothing
        if haskey(cache, e)
            return cache[e]
        end
        # Generate corresponding variable and constraint (interacts with Gurobi)
        f = MLExpr(1.0,offset+1)  
        offset += 1 # increase opt var counter
        # println("continuous ", f)
        idx = [offset] # indices of variables in constraint
        coeff = [-1.0] # coefficients in linear constraint
        terms = String[]
        for (m,c) in e
            if length(m.vars) == 1
                # push!(idx, m.vars[1])
                push!(terms, "$(c)$(m)")
            else # w = x*y
                @assert length(m.vars) == 2
                # Assumes both variables have domain [0,1]
                # Might lead to unfeasible program if this is violated
                if haskey(bilinterms,m)
                    id = bilinterms[m] # id of w
                else # add continuous variable w to problem with 0 ≤ w ≤ 1
                    offset += 1
                    id = offset # id of w
                    bilinterms[m] = id
                end
                push!(terms, "$(c)χ$(id)")
                push!(idx, id)
                # w - x ≤ 0 
                println(io, "χ$(id) - χ$(m.vars[1]) <= 0.0")
                # w - y ≤ 0 
                println(io, "χ$(id) - χ$(m.vars[2]) <= 0.0")
                # w - y - x ≥ -1
                println(io, "χ$(id) - χ$(m.vars[1]) - χ$(m.vars[1]) >= -1")
            end
            push!(coeff, c)
        end
        # println(idx, coeff)
        # for (i,c) in zip(idx, coeff)
        #     print("$c χ$(id) + ")
        # end
        println(io, "$f = ", join(terms, " + "))
        cache[e] = f
    end
    # Run variable elimination to generate constraints
    α = ADD.Terminal(MLExpr(1.0))
    for i = 1:length(ordering)
        var = ordering[i] # variable to eliminate
        # printstyled("Eliminate: ", var, "\n"; color = :red)
        α = α * reduce(*, buckets[var]; init = ADD.Terminal(MLExpr(1.0)))
        α = ADD.marginalize(α, var)        
        # Obtain copy with modified leaves and generate constraints (interacts with JUMP / Gurobi)
        α = ADD.apply(renameleaves, α)
        # For standard bucket elimination (generates tree-decomposition)
        # scope = Base.filter(n -> isa(n,ADD.Node), collect(α))
        # if isempty(scope)
        #     id = ordering[end]
        # else
        #     _, id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), scope)  
        # end   
        # push!(buckets[id], β) 
        # printstyled("-> $id\n"; color = :green)     
    end
    println(io, "Objective: ", ADD.value(α))
    ADD.value(α)
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
        @error "Something went wrong. $var is not in scope of node $(node)." 
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