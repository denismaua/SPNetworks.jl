# Parameter Learning

export
    EMParamLearner,
    initialize,
    converged,
    step


"""
    Parameter Learning Algorithm
"""
abstract type ParameterLearner end

"""
    Learn weights using the Expectation Maximization algorithm. 
"""
mutable struct EMParamLearner <: ParameterLearner
    # spn::SumProductNetwork
    # dataset::AbstractMatrix
    score::Float64     # score (loglikelihood)
    prevscore::Float64 # for checking convergence
    tolerance::Float64 # tolerance for convergence criterion
    steps::Integer   # number of learning steps (epochs)
    minimumvariance::Float64 # minimum variance for Gaussian leaves
    #EMParamLearner(spn,dataset) = new(spn,dataset,NaN,NaN,1e-8,0)
    EMParamLearner() = new(NaN,NaN,1e-8,0,0.5)
    #TODO: Hold Sparse Tensor of weight updates
end

#TODO Random initialization of EMParamLearner weights.

"""
    Random initialization of weights
"""
function initialize(spn::SumProductNetwork)
    sumnodes = filter(i -> isa(spn[i], SumNode), 1:length(spn))
    gaussiannodes = filter(i -> isa(spn[i],GaussianDistribution), 1:length(spn))
    for i in sumnodes
        ch = children(spn,i)        
        w = Random.rand(length(ch))
        spn._weights[ch,i] = w/sum(w)
    end    
    for i in gaussiannodes
        spn[i].mean = rand()
        spn[i].variance = 1.0
    end
end

""" 
    Verifies if algorithm has converged.

Convergence is defined as absolute difference of score. 
Requires at least 2 steps of optimization.
"""
converged(learner::EMParamLearner) = learner.steps < 2 ? false : abs(learner.score - learner.prevscore) < learner.tolerance


"""
    Computes the partial derivatives of the network output w.r.t. to each node at the current evaluation. Assumes network has been evaluated first in original domain.

    Returns a vector of derivatives.
"""
# function backpropagaten!(spn::SumProductNetwork)::Vector{Float64}
#     # Assumes network has been evaluted at some assignment
#     # Backpropagate derivatives
#     diff = zeros(Float64,length(spn._nodes)) 
#     diff[1] = 1.0
#     for (i,node) in enumerate(spn._nodes)
#         if isa(node, SumNode)
#             for j in spn._backward[i]
#                 diff[j] += spn._weights[j,i]*diff[i]
#             end
#         elseif isa(node, ProductNode)
#             for j in spn._backward[i]
#                 diff[j] += diff[i]*spn._values[i]/spn._values[j]
#             end
#         end
#     end
#     return diff
# end

# """
#  Computes derivatives for values propagated in log domain.
# Returns vector of derivatives.
# """
# function backpropagate(spn::SumProductNetwork)::Vector{Float64}
#     # Assumes network has been evaluted at some assignment using logpdf!
#     # Backpropagate derivatives
#     diff = zeros(Float64,length(spn)) 
#     values = similar(diff)
#     logpdf(spn,values)
#     backpropagate(spn,values,diff)
#     # diff[1] = 1.0
#     # for i = 1:length(spn)
#     #     if isa(spn[i], SumNode)
#     #         for j in children(spn,i)
#     #             diff[j] += spn._weights[j,i]*diff[i]
#     #         end
#     #     elseif isa(spn[i], ProductNode)
#     #         for j in children(spn,i) 
#     #             diff[j] += diff[i]*exp(spn._values[i]-spn._values[j])
#     #         end
#     #     end
#     # end
#     return diff
# end

"""
    backpropagate!(diff,spn,values)

 Computes derivatives for values propagated in log domain and stores results in given vector diff.
"""
function backpropagate!(diff::Vector{Float64},spn::SumProductNetwork,values::Vector{Float64})
    # Assumes network has been evaluted at some assignment using logpdf!
    # Backpropagate derivatives
    @assert length(diff) == length(spn) == length(values)
    fill!(diff,0.0) 
    @inbounds diff[1] = 1.0
    for i = 1:length(spn)
        if isa(spn[i], SumNode)
            for j in children(spn,i)
               @inbounds diff[j] += getweight(spn,i,j)*diff[i]
            end
        elseif isa(spn[i], ProductNode)
            for j in children(spn,i) 
                @inbounds diff[j] += diff[i]*exp(values[i]-values[j])
            end
        end
    end
end

# compute log derivatives (not working!)
function logbackpropagate(spn::SumProductNetwork,values::Vector{Float64},diff::Vector{Float64})
    # Assumes network has been evaluted at some assignment using logpdf!
    # Backpropagate derivatives
    @assert length(diff) == length(spn) == length(values)
    # create storage for each computed value (message from parent to child)
    from = []
    to = []
    for i = 1:length(spn)
        if isa(spn[i], SumNode) || isa(spn[i], ProductNode)
            for j in children(spn,i)
                push!(from,i)
                push!(to,j)
            end
        end
    end
    cache = sparse(to,from,ones(Float64,length(to)))
    #fill!(diff,0.0)
    logdiff = zeros(Float64,length(diff))
    for i = 1:length(spn)
        if i == 1
            diff[i] == 1.0
        else
            cache_vals = nonzeros(cache[i,:]) # incoming arc values
            offset = maximum(cache_vals)
            logdiff[i] = offset + log(sum(exp.(cache_vals.-offset)))
            diff[i] = isfinite(logdiff[i]) ? exp(logdiff[i]) : 0.0
        end
        if isa(spn[i], SumNode)
            for j in children(spn,i)
                #@inbounds diff[j] += getweight(spn,i,j)*diff[i]
                cache[j,i] = logweight(spn,i,j) + logdiff[i]
            end
        elseif isa(spn[i], ProductNode)
            for j in children(spn,i)
                #@inbounds diff[j] += diff[i]*exp(values[i]-values[j])
                cache[j,i] = logdiff[i] + values[i]-values[j]
            end
        end
    end
end

"""
 Computes derivatives for given vector of values propagated in log domain.

Returns vector of derivatives.
"""
function backpropagate(spn::SumProductNetwork,values::Vector{Float64})::Vector{Float64}
    diff = Array{Float64}(undef,length(spn))
    backpropagate!(diff,spn,values)
    return diff
end

"""
    Improvement step for learning weights using the Expectation Maximization algorithm. 

    spn.nodes[i].weights[j] = spn.nodes[i].children[j](a) * backpropagate(spn)[spn.nodes[i]]
    minimumvariance: minimum variance for Gaussian leaves
"""
function step(learner::EMParamLearner, spn::SumProductNetwork, Data::AbstractMatrix, minimumvariance::Float64 = learner.minimumvariance)
    
    # regularization constant for logweights (to avoid degenerate cases)
    τ = 1e-2
    
    numrows, numcols = size(Data)
    # @assert numcols == spn._numvars "Number of columns should match number of variables in network."
    m, n = size(spn._weights)
    weights = nonzeros(spn._weights)
    childrens = rowvals(spn._weights)
    # oldweights = fill(τ, length(weights))
    newweights = fill(τ, length(weights))
    maxweights = similar(newweights)
    score = 0.0 # data loglikelihood
    sumnodes = filter(i -> isa(spn[i], SumNode), 1:length(spn))
    gaussiannodes = filter(i -> isa(spn[i],GaussianDistribution), 1:length(spn))
    if length(gaussiannodes) > 0
        means = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
        squares = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
        denon = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
    end
    diff = zeros(Float64,length(spn))
    values = similar(diff)
    for t = 1:numrows
        datum = view(Data,t,:)
        lv = logpdf!(values,spn,datum) # propagate input Data[i,:]
        @assert isfinite(lv) "logvalue of datum $t is not finite: $lv"
        score += lv
        backpropagate!(diff,spn,values) # backpropagate derivatives
        for i in sumnodes
            # for j in children(spn,i) #spn._backward[i]
            #     oldweights[j,i] += spn._weights[j,i]*diff[i]*exp(values[j]-lv)
            # end
            for k in nzrange(spn._weights,i)
                j = childrens[k]
                #oldweights[k] += weights[k]*diff[i]*exp(values[j]-lv)
                Δ = log(weights[k]) + log(diff[i]) + values[j] - lv
                if t == 1
                    maxweights[k] = Δ
                    newweights[k] = 1.0
                else
                    if Δ > maxweights[k]
                        newweights[k] = exp(log(newweights[k])+maxweights[k]-Δ)+1.0
                        maxweights[k] = Δ
                    elseif isfinite(Δ) && isfinite(maxweights[k])
                        newweights[k] += exp(Δ-maxweights[k])
                    end
                    @assert isfinite(newweights[k]) "Infinite weight: $(newweights[k])"
                end
            end
        end
        for i in gaussiannodes
            α = diff[i]*exp(values[i]-lv)
            denon[i] += α
            means[i] += α*datum[spn[i].scope]
            squares[i] += α*datum[spn[i].scope]^2
        end
    end
    # add regularizer to avoid degenerate distributions
    newweights =  log.(newweights) .+ maxweights
    for i in sumnodes
        chval = nzrange(spn._weights,i)
        normexp!(view(newweights,chval), view(weights,chval), τ)
        if !isfinite(sum(weights[chval])) # some numerical problem occurred, set weights to uniform
            weights[chval] .= 1/length(chval)
        end
        # @assert sum(weights[chval]) ≈ 1.0 "Unormalized weight vector: $(sum(weights[chval])) | $(weights[chval])"
        # add regularizer to avoid degenerate distributions
        # Z = sum(oldweights[:,i]) + τ*length(children(spn,i))
        # for j in  children(spn,i) #spn._backward[i]
        #    spn._weights[j,i] = (oldweights[j,i]+τ)/Z
        #    # if isnan(Z)
        #    #    @warn "Not a number for weight $i -> $j"
        #    # end
        # end
    end
    for i in gaussiannodes
        spn[i].mean = means[i]/denon[i]
        spn[i].variance = squares[i]/denon[i] - (spn[i].mean)^2
        if spn[i].variance < minimumvariance
            spn[i].variance = minimumvariance
        end
    end
    learner.steps += 1
    learner.prevscore = learner.score
    learner.score = score
    return learner.score
end

