# Runs MAP Inference algorithms
using SumProductNetworks
import SumProductNetworks: leaves, issum, isprod, IndicatorFunction
import SumProductNetworks.MAP: maxproduct!, spn2bn
import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode

# spn_filename = ARGS[1]
spn_filename = "/Users/denis/code/SumProductNetworks/assets/example.pyspn.spn"

# Load SPN from spn file (assume 0-based indexing)
loadtime = @elapsed spn = SumProductNetwork(spn_filename; offset = 1)
println("Loaded in $(loadtime)s ", spn)
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
query = Set(scope(spn))

# Run max-product 
mptime = @elapsed maxproduct!(x, spn, query)
println("MaxProduct: $(spn(x)) [$(mptime)s]")

# translate spn into bayes net
variables = Dict{String,VariableNode}()
factors = Dict{String,FactorNode}()
# Add root nodes for manifest variables
for v in scope(spn)
    variables["X" * string(v)] = VariableNode(vdims[v])
end
for i = length(spn):-1:1
    node = spn[i]
    println(i, " ", node)
    if issum(node)
        # process sum node
        var = VariableNode(2)
        variables["Y"*string(i)] = var
    elseif isprod(node)
        # process product node
        var = VariableNode(2)
        variables["Y"*string(i)] = var
        factor = FactorNode(zeros(Float64, Tuple(2 for _ = 1:(length(node.children)+1)) ),VariableNode[var ; map(j -> variables["Y"*string(j)], node.children)])
        factor.factor[2:2:(end-1)] .= -Inf
        factor.factor[end-1] = -Inf
        factors[string(i)] = factor
    elseif isa(node, IndicatorFunction)
        # process leaf node
        var = VariableNode(2)
        variables["Y"*string(i)] = var
        parent = variables["X"*string(node.scope)]
        factor = FactorNode(zeros(Float64, 2, vdims[node.scope]), [var, parent])
        factor.factor[2:2:(end-1)] .= -Inf
        factor.factor[1,convert(Int,node.value)] = -Inf
        factor.factor[2,convert(Int,node.value)] = 0.0 
        factors[string(i)] = factor
    else
        @error "Unsupported node type: $(typeof(node))"
    end
end
variables