# Translates SPN to Bayesian Networks and runs Hybrid-Product Belief Propagation to obtain MAP configuration
import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode
import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, decode, setevidence!, setmapvar!

"""
    spn2bn(spn::SumProductNetwork)

Returns factor graph representing the computation graph of `spn`.
"""
function spn2bn(spn::SumProductNetwork)
    # recover dimensionality for each variable
    vdims = SumProductNetworks.vardims(spn)
    variables = Dict{String,VariableNode}()
    # sizehint!(variables,length(vdims)+length(spn))
    factors = Dict{String,FactorNode}()
    # sizehint!(factors,length(spn))
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
            # process indicator leaf node
            var = VariableNode(2)
            variables["Y"*string(i)] = var
            parent = variables["X"*string(node.scope)]
            factor = FactorNode(zeros(Float64, 2, vdims[node.scope]), [var, parent])
            # P(Y=2|parent) = { 1 if parent = node.value, 0 otherwise }
            factor.factor[2:2:end] .= -Inf
            factor.factor[1,convert(Int,node.value)] = -Inf
            factor.factor[2,convert(Int,node.value)] = 0.0 
            factors[string(i)] = factor
        elseif isa(node, CategoricalDistribution)
            # process categorical leaf node
            var = VariableNode(2)
            variables["Y"*string(i)] = var
            parent = variables["X"*string(node.scope)]
            factor = FactorNode(Array{Float64}(undef, 2, vdims[node.scope]), [var, parent])
            # P(Y=2|parent=j) = node.values[j]
            for j = 1:vdims[node.scope]
                factor.factor[2,j] = log(node.values[j])
                factor.factor[1,j] = log(1.0 - node.values[j])
            end
            factors[string(i)] = factor
        else
            @error "Unsupported node type: $(typeof(node))"
        end
    end
    # Create and return factor graph
    FactorGraph(variables, factors)
end

"""
    beliefpropagation!(x, spn, query; kwargs)

Runs mixed belief propagation algorithm with given `spn`, `query` variables and evidence `x`.

### Keyword Arguments
- `maxiterations`: positive integer value specifying maximum number of iterations [default: `10`]
- `verbose`: whether to output information during inference run [default: `true`]
- `normalize`: whether to normalize messages [defulat: `true`]
- `lowerbound`: whether to use the probability of `x` as initial lower bound [default: `false`]
- `earlystop`: early stop algorithm if residual is below that threshold [default: `0.001`]
- `rndminit`: whether to use random initialization of messages (rather than constant initialization) [default: `false`]
"""
function beliefpropagation!(x::AbstractArray{<:Real}, spn::SumProductNetwork, query; maxiterations = 10, verbose = true, normalize = true, lowerbound = false, earlystop = 0.001, rndminit = false)
    # Translate SPN into distribution-equivalent Factor Graph
    if verbose 
        fg, t, bytes, gctime, mallocs = @timed spn2bn(spn)
        @info "Generated factor graph ($(round(bytes/1048576,digits=1))MiB) in $t secs."
    else
        fg = spn2bn(spn) 
    end
    # consistency checks
    @assert length(fg.variables) == (length(spn) + length(scope(spn)))
    @assert length(fg.factors) == length(spn)

    # Initialize belief propagation
    bp = HybridBeliefPropagation(fg; rndinit = rndminit) # rndinit = true generates random initial message; = true sets all to 1.
    bp.normalize = normalize # normalize messages (sum = 1)?
    # Set evidence and query
    setevidence!(bp, "Y1", 2)
    for v in findall(isfinite,x) # filter non-marginalized variables
        if v ∉ query # filter out query variables
            setevidence!(bp, "X$v", x[v]) # remaining are evidence
        end
    end
    for v in query
        setmapvar!(bp, "X$v")
    end
    # run Inference
    if verbose
        println("Running belief propagation for $maxiterations iterations")
    end
    cpad(s, n::Integer, p=" ") = rpad(lpad(s,div(n+length(s),2),p),n,p) # for printing centered
    columns = ["it", "time (s)", "residual", "value", "best?"]
    widths = [4, 10, 10, 24, 7]
    bordercolor = :light_cyan
    headercolor = :cyan
    fieldcolor = :normal
    if verbose
        printstyled('╭', join(map(w -> repeat('─',w), widths), '┬'), '╮', '\n' ;color = bordercolor)
        for (col,w) in zip(columns,widths)
            printstyled("│"; color = bordercolor)
            printstyled(cpad(col, w); color = headercolor)
        end
        printstyled("│\n"; color = bordercolor)
        printstyled('├', join(map(w -> repeat('─',w), widths), '┼'), '┤', '\n' ;color = bordercolor)
    end
    # value of best incumbent solution
    best = -Inf
    if lowerbound
        best = spn(x)
    end
    y = copy(x) # incumbent solution
    start = time_ns()
    for it=1:maxiterations
        # if it == 1
        #     for v in query
        #         setevidence!(bp, "X$v", x[v])
        #     end
        # elseif it == 2
        #     for v in query
        #         rmevidence!(bp, "X$v")
        #     end
        # end
        # downward propagation  
        res = 0.0
        for i = length(spn):-1:1
            node = spn[i]
            # compute incoming messages from children
            if isleaf(node) 
                factor = fg.factors[string(i)]           
                update!(bp, fg.variables["X$(node.scope)"], factor)
                res = max(res, update!(bp, factor, fg.variables["Y$i"]))
            else
                factor = fg.factors[string(i)]
                for j in node.children
                    update!(bp, fg.variables["Y$j"], factor)
                end
                res = max(res, update!(bp, factor, fg.variables["Y$i"]))
            end
        end
        # upward propagation
        for i = 1:length(spn)
            node = spn[i]
            # compute outgoing messages to children
            if isleaf(node)
                factor = fg.factors[string(i)]
                update!(bp, fg.variables["Y$i"], factor)
                res = max(res, update!(bp, factor, fg.variables["X$(node.scope)"]))
            else
                factor = fg.factors[string(i)]
                update!(bp, fg.variables["Y$i"], factor)
                for j in node.children
                    res = max(res, update!(bp, factor, fg.variables["Y$j"]))
                end
            end
        end
        for i in query
            y[i] = decode(bp, "X$i")
        end
        prob = spn(y)
        if verbose
            etime = (time_ns()-start)/1e9;
            for (col,w) in zip([it, round(etime, digits=2), round(res,digits=2), prob, prob >= best ? "*" : " "],widths)
                printstyled("│"; color = bordercolor)
                printstyled(lpad(col, w-1), ' '; color = fieldcolor)
            end 
            printstyled("│\n"; color = bordercolor)
        end
        if prob > best
            best = prob
            x .= y
        end
        if res < earlystop break end # early stop
    end
    if verbose
        printstyled('╰', join(map(w -> repeat('─',w), widths), '┴'), '╯', '\n' ;color = bordercolor)
    end
    best
end