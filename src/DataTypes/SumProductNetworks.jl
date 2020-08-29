# Implements SPN data type and associated helper functions

export
    SumProductNetwork,
    scope,
    nparams


"""
    SumProductNetwork(nodes::Vector{Node})

Implements a Sum-Product Network. 
Assumes nodes are numbered topologically so that `nodes[1]` is the root (output) of the network and nodes[end] is a leaf.

# Arguments
- `nodes`: vector of nodes sorted in topological order (use sort!(spn) after creating the network if this is not the case).
"""
struct SumProductNetwork
    nodes::Vector{Node}
end

"""
    length(spn::SumProductNetwork)

Computes the number of nodes in the sum-product network `spn`.
"""
Base.length(spn::SumProductNetwork) = length(nodes(spn))

"""
    getindex(spn::SumProductNetwork, i...)

Get node of `spn` by indexes `i`.

### Examples

```julia
node = spn[1] # returns the root node
nodes = spn[1,3] # returns the root node and some other
nodes = spn[1:end] # collects all nodes
```
"""
Base.getindex(spn::SumProductNetwork, i...) = getindex(nodes(spn), i...)

"""
    firstindex(spn::SumProductNetwork)

Return the index of the root node of `spn`.
"""
Base.firstindex(spn::SumProductNetwork) = 1

"""
    lastindex(spn::SumProductNetwork)

Return the index of the last leaf node of the sum-product network `spn`.
"""
Base.lastindex(spn::SumProductNetwork) = length(spn)

"""
    Traverses network (assumes nodes are topological ordered).
"""
function Base.iterate(spn::SumProductNetwork, state = 1)
    if state > length(spn)
        return nothing
    end
    return (spn[state],state+1)
end
Base.eltype(spn::Type{SumProductNetwork}) = Node

"""
    sort!(spn::SumProductNetwork)

Sort nodes in topological order (with ties broken by breadth-first order) and modify node ids accordingly.
Returns the permutation applied.
"""
function sort!(spn::SumProductNetwork)    
    # First compute the number of parents for each node
   pa = zeros(length(spn))
   for (i,n) in enumerate(spn)
        if !isleaf(n)
            for j in n.children            
                pa[j] += 1
            end
        end
    end
    @assert count(isequal(0), pa) == 1 "SumProductNetworks has more than one parentless node"
    root = findfirst(isequal(0),pa) # root is the single parentless node
    # Kanh's algorithm: collect node ids in topological BFS order
    open = Vector{Int}()
    # visited = Set{Int}()
    closed = Vector{Int}() # topo bfs order
    push!(open, root) # enqueue root node 
    while !isempty(open)
        n = popfirst!(open) # dequeue node
        # push!(visited, n)
        push!(closed, n)
        if !isleaf(spn[n])
            # append!(open, ch for ch in spn[n].children if !in(ch, visited) && !in(ch, open))
            for j in spn[n].children
                pa[j] -= 1
                if pa[j] == 0
                    push!(open, j)
                end
            end
        end
    end
    @assert length(closed) == length(spn)
    inverse = similar(closed) # inverse mapping
    for i=1:length(closed)
        inverse[closed[i]] = i
    end
    # permute nodes according to closed
    permute!(spn.nodes,closed) # is this faster than spn.nodes .= spn.nodes[closed]? 
    # now fix ids of children
    for i=1:length(spn)
        if !isleaf(spn.nodes[i])
            for (j,ch) in enumerate(spn.nodes[i].children)
                spn.nodes[i].children[j] = inverse[ch]
            end
        end
    end
    closed
end
"""
    nodes(spn::SumProductNetwork)

Collects the list of nodes in `spn`.
"""
@inline nodes(spn::SumProductNetwork) = spn.nodes


"""
Select nodes by topology
"""
@inline leaves(spn::SumProductNetwork) = filter(n -> isa(n, LeafNode), nodes(spn)) 
@inline sumnodes(spn::SumProductNetwork) = filter(n -> isa(n, SumNode), nodes(spn)) 
@inline productnodes(spn::SumProductNetwork) = filter(n -> isa(n, ProductNode), nodes(spn)) 
@inline root(spn::SumProductNetwork) = @inbounds spn.nodes[1]
#TODO #variables(spn::SumProductNetwork) = collect(1:spn._numvars)
@inline children(spn::SumProductNetwork,n) = @inbounds spn.nodes[n].children
"""
Return vector of weights associate to outgoing edges of (sum) node n.
"""
@inline weights(spn::SumProductNetwork, n) = @inbounds spn.nodes[n].weights
# """
# Return the value of the weight associate to edge from `i` to `j`
# """
#@inline getweight(spn::SumProductNetwork, i, j) = @inbounds spn.nodes[n].weights[j]
# """
# Return the log-value of the weight associate to edge from `i` to `j`
# """
#@inline logweight(spn::SumProductNetwork, i, j) = log(getweight(spn,i,j))

"""
    nparams(spn::SumProductNetwork)

Computes the number of parameters in the network `spn`.
"""
function nparams(spn::SumProductNetwork)
    numparams = 0
    for i = 1:length(spn)
        if issum(spn[i])
            numparams += length(children(spn,i))
        elseif isa(spn[i],CategoricalDistribution)
            numparams += length(spn[i].values)
        elseif isa(spn[i],GaussianDistribution)
            numparams += 2
        end
    end
    numparams
end

"""
    vardims(spn::SumProductNetwork)

Returns a dictionary mapping each variable index to its dimension (no. of values).
Assigns dimension = -1 for continuous variables.
"""
function vardims(spn::SumProductNetwork)
    vdims = Dict{Int,Int}()
    for node in leaves(spn)
        if isa(node, IndicatorFunction)
            dim = get(vdims, node.scope, 0)
            vdims[node.scope] = max(dim,convert(Int,node.value))
        elseif isa(node, CategoricalDistribution)
            vdims[node.scope] = length(node.values)
        elseif isa(node, GaussianDistribution)
            vdims[node.scope] = -1
        end
    end
    vdims
end

"""
    scope(spn)

Returns the scope of network `spn`, given by the scope of its root node.
"""
scope(spn::SumProductNetwork)::AbstractVector = unique(collect(map(n->scope(n), leaves(spn))))  

"""
    scopes(spn::SumProductNetwork)

Returns an array of scopes for every node in the `spn` (ordered by their index).
"""
function scopes(spn::SumProductNetwork)
    sclist = Array{Array{Int}}(undef, length(spn))
    for i = length(spn):-1:1
        node = spn[i]
        if isleaf(node)
            sclist[i] = Int[node.scope]
        elseif issum(node) # assume completeness
            sclist[i] = copy(sclist[node.children[1]])
        else # can probably be done more efficiently
            sclist[i] = Base.reduce(union, map(j -> sclist[j], node.children)) 
        end
    end
    sclist
end


"""
    project(spn::SumProductNetwork,query::AbstractSet,evidence::AbstractVector)

Returns the projection of a _normalized_ `spn` onto the scope of `query` by removing marginalized subnetworks of marginalized variables and reducing subnetworks with fixed `evidence`.
Marginalized variables are the ones that are not in `query` and are assigned `NaN` in `evidence`. 
The projected spn assigns the same values to configurations that agree on evidence and marginalized variables w.r.t. to `evidence`.
The scope of the generated network contains query and evidence variables, but not marginalized variables.
"""
function project(spn::SumProductNetwork,query::AbstractSet,evidence::AbstractVector)
    nodes = Dict{UInt,Node}()
    # evaluate network to collect node values
    vals = Array{Float64}(undef, length(spn))
    SumProductNetworks.logpdf!(vals, spn, evidence);
    # collect marginalized variables
    marginalized = Set(Base.filter(i -> (isnan(evidence[i]) && (i ∉ query)), 1:length(evidence)))
    # collect evidence variables
    evidvars = Set(Base.filter(i -> (!isnan(evidence[i]) && (i ∉ query)), 1:length(evidence)))
    println(marginalized)
    nscopes = scopes(spn)
    newid = length(spn) + 1 # unused id for new node
    stack = UInt[ 1 ]    
    while !isempty(stack)
        n = pop!(stack)
        node = spn[n]
        if isprod(node)
            children = UInt[]
            for ch in node.children
                if !isempty(nscopes[ch] ∩ query)
                    # subnetwork contains query variables, keep it
                    push!(children, ch)
                    push!(stack, ch)
                else # Replace node with subnetwork of equal value
                    for e in (evidvars ∩ nscopes[ch])
                        if !haskey(nodes, ch) # only if we haven't already done this
                            nodes[ch] = SumNode([newid, newid+1], [exp(vals[ch]), 1.0-exp(vals[ch])])
                            nodes[newid] = IndicatorFunction(e, evidence[e])
                            nodes[newid+1] = IndicatorFunction(e, NaN) # arbitrary different value
                            newid += 2                                                        
                        end
                        push!(children, ch)
                    end
                end
            end
            nodes[n] = ProductNode(children)
        else 
            if issum(node)
                append!(stack, node.children)
            end
            nodes[n] = deepcopy(node)
        end
    end
    # Reassign indices so that the become contiguous    
    # Sorted list of remaining node ids -- position in list gives new index
    nodeid = Base.sort!(collect(keys(nodes)))
    idmap = Dict{UInt,UInt}()
    for (newid, oldid) in enumerate(nodeid)
        idmap[oldid] = newid
    end
    # Now remap ids of children nodes
    for node in values(nodes)
        if !isleaf(node)
            for (i, ch) in enumerate(node.children)
                node.children[i] = idmap[ch]
            end
        end
    end
    # println(idmap)
    spn = SumProductNetwork([ nodes[i] for i in nodeid ])
    sort!(spn) # ensure nodes are topologically sorted (with ties broken by bfs-order)
    spn    
end