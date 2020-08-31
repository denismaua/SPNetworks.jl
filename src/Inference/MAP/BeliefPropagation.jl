# Translates SPN to Bayesian Networks and runs Hybrid-Product Belief Propagation to obtain MAP configuration
import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode
import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, decode, setevidence!, setmapvar!
import GraphicalModels.MessagePassing: rmevidence!
# TODO: dichotomize nodes (make max no. of children for inner nodes = 2) 

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
            if length(node.children) > 4
                @warn "node $i indegree is too large: $(length(node.children)). It is highly recommend to split nodes before running this. Try running binarize! first."
            end
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
            if length(node.children) > 4
                @warn "node $i indegree is too large: $(length(node.children)). It is highly recommend to split nodes before running this. Try running binarize! first."
            end
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
- `warmstart`: whether to use the configuration of query variables in `x` to initilize messages [default: `false`]
- `earlystop`: early stop algorithm if residual is below that threshold [default: `0.001`]
- `rndminit`: whether to use random initialization of messages (rather than constant initialization) [default: `false`]
"""
function beliefpropagation!(x::AbstractArray{<:Real}, spn::SumProductNetwork, query; 
        maxiterations = 10, 
        verbose = true, 
        normalize = true, 
        lowerbound = false, 
        earlystop = 0.001, 
        rndminit = false,
        warmstart = false )
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
    # for computing logpdf of solution
    values = Vector{Float64}(undef,length(spn))
    # value of best incumbent solution
    best = -Inf
    if lowerbound
        # best = spn(x)
        best = logpdf!(values,spn,x)
    end
    if warmstart
        # bias messages towards initial solution
        for v in query
            var = fg.variables["X$v"]
            for ne in var.neighbors
                bp.messages[(var,ne)] .+= 1
            end
        end
    end
    y = copy(x) # incumbent solution
    start = time_ns()
    for it=1:maxiterations
        # if warmstart
        #     if it == 1
        #         for v in query
        #             setevidence!(bp, "X$v", x[v])
        #         end
        #     elseif it == 2
        #         for v in query
        #             rmevidence!(bp, "X$v")
        #         end
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
        # prob = spn(y)
        prob = logpdf!(values,spn,y)
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
    exp(best)
end

"""
    treebeliefpropagation!(x::AbstractArray{<:Real}, spn::SumProductNetwork, query; kwargs)

Speciliazed hybrid belief propagation for tree-shaped SPNs where innner nodes have two children, leaves are indicators.
"""
function treebeliefpropagation!(x::AbstractArray{<:Real}, spn::SumProductNetwork, query; 
    maxiterations = 10, 
    verbose = true, 
    normalize = true, 
    lowerbound = false, 
    earlystop = 0.001, 
    rndminit = false,
    warmstart = false  )
    # For conveniency
    leaves = filter(i -> isleaf(spn[i]), 1:length(spn))
    gates = filter(i -> !isleaf(spn[i]), 1:length(spn))
    nleaves, ngates = length(leaves), length(gates)
    evidence = setdiff(Set(Base.filter(i -> isfinite(x[i]), 1:length(x))), query)
    # Compute log( exp(x1)/(exp(x1)+exp(x2)) )
    lognorm(x1, x2) = x1 >= x2 ? -log1p(exp(x2 - x1)) : x1 - x2 - log1p(exp(x1 - x2))
    # Messages
    π = Dict{Int,Float64}()  # upstream messages from node n
    π_v = Dict{Tuple{Int, Int}, Array{Float64}}()  # upstream messages from manifest variables
    π_i = Dict{Tuple{Int, Int}, Float64}()  # upstream messages from indicator nodes
    λ = Dict{Int, Float64}()  # downstream messages
    λ_i = Dict{Tuple{Int, Int}, Float64}()  # downstream messages to indicator nodes
    λ_v = Dict{Tuple{Int, Int}, Array{Float64}}()  # downstream messages to manifest variables
    out = Dict{Int,Int}()  # parent of nodes 
    out_i = Dict{Int,Array{Int}}() # parents of indicator nodes
    out_v = Dict{Int,Array{Int}}() # parents of manifest variables
    bel = Dict{Int, Array{Float64}}()  # beliefs for manifest variables
    ### Initialization ################################################################
    vdims = vardims(spn)
    for (var, ncat) in vdims
        # Initialize beliefs
        # evidence or warm start
        if var in evidence || (warmstart && (var in query) && !isnan(x[var]))
            bel[var] = zeros(Float64, ncat)
            bel[var][Int(x[var])] = 1.0
        else
            bel[var] = ones(Float64, ncat)/ncat
        end
        out_v[var] = []
    end
    for n in leaves
        node = spn[n]
        @assert isa(node, IndicatorFunction)
        ncat = vdims[node.scope] 
        λ_v[(n, node.scope)] = ones(Float64, ncat)/ncat
        π_v[(node.scope, n)] = ones(Float64, ncat)/ncat
        push!(out_v[node.scope], n)
        out_i[n] = []
    end
    out[1] = -1 # root/output node has no parents
    for n in gates
        node = spn[n]
        @assert length(node.children) == 2
        for ch in node.children                    
            if isa(spn[ch],IndicatorFunction)
                push!(out_i[ch], n)
                π_i[(ch,n)] = 0.5
                λ_i[(n,ch)] = 0.5
            else
                λ[ch] = 0.5
                π[ch] = 0.5
                out[ch] = n 
            end
        end 
    end
    λ[1] = 1.0 # Root/output node receives "evidence"
    ### Run message passing ###########################################################
    best = -Inf
    y = copy(x) # candidate solution
    for iteration = 1:maxiterations
        ### upstream propagation ##########################################################
        ## compute messages π[N -> P] -- assumes incoming messages are normalized
        # compute messages from variables to indicator nodes
        for (edge, message) in π_v
            var, n = edge
            if var in evidence
                # variable is evidence
                fill!(message, 0.0)
                message[Int(x[var])] = 1.0
            else
                Z = 0.0
                for j = 1:vdims[var]
                    message[j] = prod(λ_v[(n, var)][j] for k in out_v[var] if k ≠ n)
                    Z += message[j]
                end
                @assert Z > 0
                # normalize messages (seems to be important only for marginalized variables)
                message ./= Z 
            end
            # print("root",v,node.assignment,values,Z)
        end
        # compute messages from indicator nodes
        for n in leaves
            node = spn[n]
            vid = node.scope
            for pa in out_i[n]
                @assert length(out_i[n]) > 0
                if x[vid] == NaN && (var ∉ query) # marginalized
                    π1 = π_v[(vid, n)][Int(node.value)]
                    for opa in out_i[n] 
                        if opa ≠ pa
                            π1 *= λ_i[(opa, n)]
                        end
                    end
                    π0 = (1 - π_v[(vid, n)][Int(node.value)])
                    for opa in out_i[n] 
                        if opa ≠ pa
                            π0 *= 1.0 - λ_i[(opa, n)]
                        end
                    end                    
                else
                    maxbel = maximum(bel[vid])
                    # TODO: handle evidence more efficiently
                    if bel[vid][Int(node.value)] == maxbel
                        π1 = π_v[(vid, n)][Int(node.value)] 
                        for opa in out_i[n] 
                            if opa ≠ pa
                                π1 *= λ_i[(opa, n)]
                            end
                        end
                    else
                        π1 = 0.0
                    end
                    π0 = 0.0
                    for j = 1:vdims[vid] 
                        if j ≠ node.value && bel[vid][j] == maxbel
                            π0 += π_v[(vid, n)][j]
                        end
                    end
                    for opa in out_i[n] 
                        if opa ≠ pa
                            π0 *= 1.0 - λ_i[(opa, n)]
                        end
                    end
                end
                @assert π0 + π1 > 0
                π_i[(n,pa)] = π1 / (π1 + π0)  # normalization
            end
        end
        # compute remaining messages
        for n in reverse(gates)
            node = spn[n]
            if issum(node)
                π[n] = 0.0
                for j = 1:2
                    if isleaf(spn[node.children[j]])
                        π[n] += node.weights[j] * π_i[(node.children[j],n)]
                    else
                        π[n] += node.weights[j] * π[node.children[j]]
                    end
                end
            else  # Product
                π[n] = 1.0
                for j = 1:2
                    if isleaf(spn[node.children[j]])
                        π[n] *= π_i[(node.children[j],n)]
                    else
                        π[n] *= π[node.children[j]]
                    end
                end
            end 
        end
        ### Upward propagation ##########################################################
        for n in gates
            # compute messages λ[N -> C] -- assumes incoming messages are normalized
            node = spn[n]
            ch1, ch2 = node.children
            λp = λ[n]
            if isa(spn[ch2],IndicatorFunction)
                π2 =  π_i[(ch2,n)]
            else    
                π2 =  π[ch2]
            end
            # left child
            if issum(node)
                λ1 = (1 - λp) * node.weights[2] * (1 - π2) + λp * (node.weights[1] * (1 - π2) + π2)
                λ0 = (1 - λp) * (1 - π2 + node.weights[1] * π2) + λp * node.weights[2] * π2
            else # product node
                λ1 = (1.0 - λp) * (1.0 - π2) + λp * π2
                λ0 = 1 - λp
            end
            @assert λ0 + λ1 > 0
            if isa(spn[ch1],IndicatorFunction)
                λ_i[(n,ch1)] = λ1 / (λ0 + λ1)
            else
                λ[ch1] = λ1 / (λ0 + λ1)
            end
            if isa(spn[ch1],IndicatorFunction)
                π1 =  π_i[(ch1,n)]
            else    
                π1 =  π[ch1]
            end                
            # right child
            if issum(node)
                λ1 = (1 - λp) * node.weights[1] * (1 - π1) + λp * (node.weights[2] * (1 - π1) + π1)
                λ0 = (1 - λp) * (1 - π1 + node.weights[2] * π1) + λp * node.weights[1] * π1
            else # product
                λ1 = (1.0 - λp) * (1.0 - π1) + λp * π1
                λ0 = 1 - λp
            end
            @assert λ0 + λ1 > 0
            if isa(spn[ch1],IndicatorFunction)
                λ_i[(n,ch2)] = λ1 / (λ0 + λ1)    
            else
                λ[ch2] = λ1 / (λ0 + λ1)    
            end
        end
        # compute messages from indicator nodes to root nodes
        for (p, message) in λ_v
            n, v = p
            λ1 = prod(λ_i[(pa, n)] for pa in out_i[n])
            λ0 = prod(1.0 - λ_i[(pa, n)] for pa in out_i[n])
            Z = λ1 + λ0 * (vdims[v] - 1) # normalization constant
            @assert Z > 0
            message .= (λ0 / Z)
            message[Int(spn[n].value)] = λ1 / Z
        end
        ### Compute beliefs ##########################################################
        for var in keys(vdims)
            if var in evidence
                fill!(bel[var], 0.0)
                bel[var][Int(x[var])] = 1.0
            else
                Z = 0.0
                for j = 1:vdims[var] 
                    b = prod(λ_v[(n, var)][j] for n in out_v[var])
                    Z += b
                    bel[var][j] = b
                end
                @assert Z > 0
                bel[var] ./= Z
                if var in query
                    # decode solution
                    m, y[var] = findmax(bel[var])
                    @assert isfinite(m) && m > 0
                end
            end
        end
        # compute value at root/output
        prob = π[1]
        value = spn(y)
        if verbose
            printstyled("[$iteration/$maxiterations] "; color = :blue)
            printstyled("Root Prob: "; color = :light_cyan)
            print(prob)
            printstyled("\t Value: "; color = :light_cyan) 
            println(value)
        end
        if value > best
            # found improving solution
            best = value
            x .= y
        end
    end
    best
end