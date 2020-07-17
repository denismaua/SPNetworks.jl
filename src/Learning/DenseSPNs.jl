"""
# DenseSPNs.jl

Implements dense SPNs for representing images

"""

"""
## Region Graphs 

From "Dennis, A. and Ventura, D. Learning the architecture of sum-product networks using clustering on variables. 
In Advances in Neural Information Processing Systems 25, 2012":

A *region graph* is a rooted DAG consisting of region nodes and partition nodes. 
The root node is a region node. 
Partition nodes are restricted to being the children of region nodes and vice versa. 
Region and partition nodes have scopes just like nodes in an SPN. 
"""

import Base: hash, isequal


export
    RegionGraphNode,
    RegionNode,
    PartitionNode,
    regionGraph,
    buildDenseSPN

abstract type RegionGraphNode end

# Represents rectangular region
# Assume x1,x2,y1,y2 are positive
struct RegionNode  <: RegionGraphNode 
    children::Vector{RegionGraphNode}
    x1::Int64
    x2::Int64
    y1::Int64
    y2::Int64
    RegionNode(x1,x2,y1,y2) = new(Vector{PartitionNode}(),x1,x2,y1,y2)
end

# Represents decomposition of regions
struct PartitionNode <: RegionGraphNode
    children::Vector{RegionGraphNode}
    x1::Int64
    x2::Int64
    y1::Int64
    y2::Int64
end

Coordinates = Tuple{Int64,Int64,Int64,Int64}

# The sum and product of two numbers a, b satisfy the equation
# x^2 - (a+b)x + (ab) = 0
# Since the quadratic equation admits only two roots, a and b must be unique
# Hence, if we assume that 0 < a < b, (a+b) + ab == (x+y) + xy iff a=b and x=y
# We can enforce that by using a and b'=a+b (as a > 0)
#Base.hash(r::RegionGraphNode) = 2*(r.x1 + r.x2 + r.x1*r.x2) + r.y1 + r.y2 + r.y1*r.y2 + (r.x1 + r.x2 + r.x1*r.x2)*(r.x1 + r.x2 + r.x1*r.x2+r.y1 + r.y2 + r.y1*r.y2)
#Base.isequal(r1::RegionGraphNode, r2::RegionGraphNode) = (r1.x1 == r2.x1 && r1.x2 == r2.x2 && r1.y1 == r2.y1 && r1.y2 == r2.y2)

# Build region graph on [x1,x2]x[y1,y2], with numparts decompositions per split
function regionGraph(x1::Int64,x2::Int64,y1::Int64,y2::Int64,numparts)
    cache = Dict{Coordinates,RegionNode}()
    r = getOrCreateRegionNode(x1,x2,y1,y2,numparts,cache)
    #println(length(cache), " in cache")
    return r
end

function getOrCreateRegionNode(x1::Int64,x2::Int64,y1::Int64,y2::Int64,numparts::Int64,cache::Dict{Coordinates,RegionNode})
    # root node is region [x1,x2]x[y1,y2] with 3 decompositions
    #spaces = repeat(' ',depth)
    #println("$(hash(r)): $spaces [$(x1),$(x2)]x[$(y1),$(y2)]")
    # if region is pixel, create new node and return it
    if x1 == x2 && y1 == y2
        return RegionNode(x1,x2,y1,y2)
    end
    # if region has already been created 
    # return it (and do not decompose it nor duplicated it)
    if haskey(cache,(x1,x2,y1,y2))
        return cache[x1,x2,y1,y2]
    end
    r = RegionNode(x1,x2,y1,y2)
    # create binary decompositions and recur
    # try splitting horizontally if possible
    if y1 < y2
        for i = 1:numparts
            y = rand(y1:y2) # split at y
            if y == y2
                r1 = getOrCreateRegionNode(x1,x2,y1,y-1,numparts,cache)
                r2 = getOrCreateRegionNode(x1,x2,y2,y2,numparts,cache)
            else
                r1 = getOrCreateRegionNode(x1,x2,y1,y,numparts,cache)
                r2 = getOrCreateRegionNode(x1,x2,y+1,y2,numparts,cache)
            end
            part = PartitionNode([r1,r2],x1,x2,y1,y2)
            push!(r.children,part)
        end
    # otherwise split vertically
    elseif x1 < x2
        for i = 1:numparts
            x = rand(x1:x2) # split at
            if x == x2
                r1 = getOrCreateRegionNode(x1,x-1,y1,y2,numparts,cache)
                r2 = getOrCreateRegionNode(x2,x2,y1,y2,numparts,cache)
            else
                r1 = getOrCreateRegionNode(x1,x,y1,y2,numparts,cache)
                r2 = getOrCreateRegionNode(x+1,x2,y1,y2,numparts,cache)
            end
            part = PartitionNode([r1,r2],x1,x2,y1,y2)
            push!(r.children,part)
        end
    end
    cache[x1,x2,y1,y2] = r
    return r
end

function Base.show(io::IO, n::RegionNode)
  show(io, n, 0)
end

function Base.show(io::IO, n::RegionNode, h::Int)
  println(io, " "^h, "(+ [$(n.x1),$(n.x2)]x[$(n.y1),$(n.y2)] $(objectid(n))")
  for ch in n.children
    show(io, ch, h+1)
  end
  println(io, " "^h, ")")
end

function Base.show(io::IO, n::PartitionNode, h::Int)
  println(io, " "^h, "(* ")
  for ch in n.children
    show(io, ch, h+1)
  end
  println(io, " "^h, ")")
end

"""
    Build Discrete Dense SPN from Region Graph.

Takes a region [x1,x2]x[y1,y2], a number numcat of values per pixel, and
a number numpart of decompositions per split.
Use numcat=0 to generate Gaussian leaves.
"""
function buildDenseSPN(x1::Int64,x2::Int64,y1::Int64,y2::Int64,numcat::Int64,numparts=2)
    r = regionGraph(x1::Int64,x2::Int64,y1::Int64,y2::Int64,numparts)
    # obtain topological order of nodes using Kahn's algorithm
    nodes::Vector{Node} = []
    backward::Vector{Vector{Int64}} = Vector() # indexed by [src, dst]
    n2i = Dict{RegionGraphNode,Int64}() # map region graph node into spn node id
    i2n = Dict{Int64,RegionGraphNode}() # inverse map
    l2n = Dict{Int64,Int64}() # map variable indicator to spn node id
    visited = Set{RegionGraphNode}()
    # first compute node in-degree
    indegree = Dict{RegionGraphNode,Int64}(r => 0)
    queue = RegionGraphNode[r]
    while length(queue) > 0
        n = pop!(queue)
        push!(visited,n)
        for ch in n.children
            if haskey(indegree, ch)
                indegree[ch] += 1
            else
                indegree[ch] = 1
            end
        end
        append!(queue, ch for ch in n.children if !in(ch, visited) && !in(ch, queue))                                                
    end
    #@info "#region nodes $(length(visited))"
    tovisit = RegionGraphNode[r]
    # now obtain indices in topo order
    i = 0
    while length(tovisit) > 0
        i += 1
        n = pop!(tovisit)
        if isa(n, RegionNode)
            push!(nodes,SumNode())
        else
            push!(nodes,ProductNode())                        
        end
        push!(backward, Int64[])
        n2i[n] = i
        i2n[i] = n
        for ch in n.children
            indegree[ch] -= 1
            #@assert indegree[ch] >= 0
            if indegree[ch] == 0
                push!(tovisit, ch)
            end
        end
    end
    @assert length(nodes) == length(keys(indegree)) "Nodes and Indegree size mismatch: $(length(nodes)) $(length(keys(indegree)))"
    @assert length(keys(n2i)) == size(backward,1) "n2i and backward size mismatch"
    # create leaves
    for i = 1:((r.x2-r.x1+1)*(r.y2-r.y1+1))
        l2n[i] = length(nodes)+1
        if numcat > 0
            for k=1:numcat
                values = zeros(Int64, numcat)
                values[k] = 1
                push!(nodes,CategoricalDistribution(i, values))      
            end
        else
            push!(nodes,GaussianDistribution(i,0.0,1.0))
        end
    end
    # now obtain DAG
    from = Int64[]
    to = Int64[]
    for i = 1:length(backward)
        n = i2n[i]
        if length(n.children) > 0
            append!( backward[i], n2i[ch] for ch in n.children )
        else
            var = n.x1 + (r.x2-r.x1+1)*(n.y1-1) 
            if numcat > 0
                append!( backward[i], [l2n[var]+k-1 for k=1:numcat])
            else
                append!( backward[i], [l2n[var]])
            end
        end
        if isa(nodes[i],SumNode)
            for ch in backward[i]
                push!(to, ch)
                push!(from, i)      
            end
        end
    end
    weights = sparse(to, from, ones(Float64, length(to)))
    # start with arbitrary weights
    sumnodes = filter(i -> isa(nodes[i], SumNode), 1:length(nodes))
    for i in sumnodes
        ch = backward[i]    
        w = Random.rand(length(ch))
        weights[ch,i] = w/sum(w)
    end   
    return SumProductNetwork{Int64}(nodes,backward,weights)
end

