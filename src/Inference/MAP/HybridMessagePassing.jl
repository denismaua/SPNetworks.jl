# Translates SPN to Bayesian Networks and runs Hybrid Message Passing There
import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode
"""
    spn2bn(spn::SumProductNetwork)

Returns factor graph representing the computation graph of `spn`.
"""
function spn2bn(spn::SumProductNetwork)
    # recover dimensionality for each variable
    vdims = SumProductNetworks.vardims(spn)
    variables = Dict{String,VariableNode}()
    factors = Dict{String,FactorNode}()
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
                prob = min(1.0,mapreduce(j -> node.weights[j]*(z[j]-1), +, 1:length(z)))
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
    # Create and return factor graph
    FactorGraph(variables, factors)
end