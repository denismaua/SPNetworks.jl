# Runs MAP Inference algorithms
using Printf # for pretty printing

using SumProductNetworks
import SumProductNetworks: leaves, isleaf, issum, isprod, IndicatorFunction
import SumProductNetworks.MAP: maxproduct!, spn2bn
import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode
import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, decode, marginal, setevidence!, setmapvar!

# spn_filename = ARGS[1]
spn_filename = "/Users/denis/code/SumProductNetworks/assets/example.pyspn.spn"
# spn_filename = "/Users/denis/learned-spns/spambase/spambase.uai"

# Load SPN from spn file (assume 0-based indexing)
loadtime = @elapsed spn = SumProductNetwork(spn_filename; offset = 1)
println("Loaded in $(loadtime)s ", summary(spn))
# recover dimensionality for each variable
vdims = Dict{Int,Int}()
for node in leaves(spn)
    @assert isa(node, IndicatorFunction)
    dim = get(vdims, node.scope, 0)
    vdims[node.scope] = max(dim,convert(Int,node.value))
end
nvars = length(scope(spn))
@assert nvars == length(vdims) "$nvars ≠ $(length(vdims))"

# Load evidence and query variables
# TODO
x = ones(Float64, nvars)
# query = Set(scope(spn))

# Run max-product 
# mptime = @elapsed maxproduct!(x, spn, query)
# println("MaxProduct: $(spn(x)) [$(mptime)s]")

# translate spn into bayes net
variables = Dict{String,VariableNode}()
factors = Dict{String,FactorNode}()
# schedule = Vector{Tuple{FGNode,FGNode}}() # downward message scheduling 
# Add root nodes for manifest variables
for (v,d) in vdims
    variables["X" * string(v)] = VariableNode(d)
end
for i = length(spn):-1:1
    node = spn[i]
    # println(i, " ", node)
    if issum(node)
        # process sum node
        var = VariableNode(2)
        variables["Y"*string(i)] = var
        factor = FactorNode(
            Array{Float64}(undef, Tuple(2 for _ = 1:(length(node.children)+1)) ), # factor
            VariableNode[var ; map(j -> variables["Y"*string(j)], node.children)] # neighbors
            )
        # P(Y=2|parents=z) = sum( node.weights[i] | z[i] = 2 )
        for z in CartesianIndices(Tuple(2 for _ = 1:length(node.children)))
            prob = mapreduce(j -> node.weights[j]*(z[j]-1), +, 1:length(z))
            factor.factor[2,z] = log(prob)
            factor.factor[1,z] = log(1 - prob)
        end
        factors[string(i)] = factor
        # display(factor.neighbors)
    elseif isprod(node)
        # process product node
        var = VariableNode(2)
        variables["Y"*string(i)] = var
        factor = FactorNode(
            zeros(Float64, Tuple(2 for _ = 1:(length(node.children)+1)) ), # factor
            VariableNode[var ; map(j -> variables["Y"*string(j)], node.children)] # neighbors
            )
        # P(Y=2|parents) = { 1 if all parents = 1, 0 otherwise }
        factor.factor[2:2:(end-1)] .= -Inf
        factor.factor[end-1] = -Inf
        factors[string(i)] = factor
    elseif isa(node, IndicatorFunction)
        # process leaf node
        var = VariableNode(2)
        variables["Y"*string(i)] = var
        parent = variables["X"*string(node.scope)]
        factor = FactorNode(zeros(Float64, 2, vdims[node.scope]), [var, parent])
        # P(Y=2|parent) = { 1 if parent = node.value, 0 otherwise }
        factor.factor[2:2:end] .= -Inf
        factor.factor[1,convert(Int,node.value)] = -Inf
        factor.factor[2,convert(Int,node.value)] = 0.0 
        factors[string(i)] = factor
    else
        @error "Unsupported node type: $(typeof(node))"
    end
end
@assert length(variables) == (length(spn) + nvars)
@assert length(factors) == length(spn)
# Create Factor Graph
fg = FactorGraph(variables, factors)
# Initialize belief propagation
bp = HybridBeliefPropagation(fg; rndinit = false)
# bp.normalize = true
# setevidence!(bp, "X1", 1)
# setevidence!(bp, "X2", 2)
setevidence!(bp, "Y1", 2)
# run Inference
printstyled("╭────┬──────────┬────────┬────────────────────┬───────╮\n"; color = :blue)
printstyled("│ it │   time   │   res  │        value       │ best? │\n"; color = :blue) 
printstyled("├────┼──────────┼────────┼────────────────────┼───────┤\n"; color = :blue)
start = time_ns()
for it=1:10
    # res = update!(bp)  
    # downward propagation  
    res = 0.0
    for i = length(spn):-1:1
        node = spn[i]
        # print("$i ", variables["Y$i"])
        # compute incoming messages from children
        if isleaf(node) 
            factor = factors[string(i)]           
            update!(bp, variables["X$(node.scope)"], factor)
            res = max(res, update!(bp, factor, variables["Y$i"]))
        else
            factor = factors[string(i)]
            for j in node.children
                update!(bp, variables["Y$j"], factor)
            end
            res = max(res, update!(bp, factor, variables["Y$i"]))
        end
        # println( marginal(bp, "Y$i") )
    end
    # upward propagation
    for i = 1:length(spn)
        node = spn[i]
        # compute outgoing messages to children
        if isleaf(node)
            factor = factors[string(i)]
            update!(bp, variables["Y$i"], factor)
            res = max(res, update!(bp, factor, variables["X$(node.scope)"]))
        else
            factor = factors[string(i)]
            update!(bp, variables["Y$i"], factor)
            for j in node.children
                res = max(res, update!(bp, factor, variables["Y$j"]))
            end
        end
    end
    etime = (time_ns()-start)/1e9;
    # prob = marginal(bp,"Y1")[2]
    # println("$it \t [$(etime)s] \t $res \t $prob")
    for i = 1:nvars
        x[i] = decode(bp, "X$i")
    end
    prob = spn(x)
    # println("$it [", round(etime,digits=3), "s] ", res, " ", prob)    
    printstyled("│ ", @sprintf("%2d", it), " │ ", @sprintf("%8.2f",etime), " │ "; color = :blue)
    @printf "%4.4f" res
    printstyled(" │ "; color = :blue) 
    @printf "%1.16f" prob
    # printstyled(IOContext(stdout, :compact => true, :limit => true), res, " | ", prob, " │ ") 
    # printstyled(sprint(print, it; context = :color => :blue))
    printstyled(" │       │\n"; color = :blue) 
    
end
printstyled("╰────┴──────────┴────────┴────────────────────┴───────╯\n"; color = :blue)

for i = 1:nvars
    x[i] = decode(bp, "X$i")
    print(x[i], " ")
end
println( spn(x) )