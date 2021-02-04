# Parameter learning by Batch Expectation-Maximimzation

"""
Learn weights using the Expectation Maximization algorithm. 
"""
mutable struct EMParamLearner <: ParameterLearner
spn::SumProductNetwork
cache::SumProductNetwork
# dataset::AbstractMatrix
score::Float64     # score (loglikelihood)
prevscore::Float64 # for checking convergence
tolerance::Float64 # tolerance for convergence criterion
steps::Integer   # number of learning steps (epochs)
minimumvariance::Float64 # minimum variance for Gaussian leaves
EMParamLearner(spn::SumProductNetwork) = new(spn,deepcopy(spn),NaN,NaN,1e-3,0,0.5)
#TODO: use sparse tensor of weight updates
end

"""
Random initialization of weights
"""
function initialize(learner::EMParamLearner) #spn::SumProductNetwork)
    spn = learner.spn
    sumnodes = filter(i -> isa(spn[i], SumNode), 1:length(spn))
    for i in sumnodes
        #@inbounds ch = spn[i].children  # children(spn,i)        
        @inbounds Random.rand!(spn[i].weights)
        @inbounds spn[i].weights ./= sum(spn[i].weights) 
        @assert sum(spn[i].weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(spn[i].weights)) | $(spn[i].weights)"
    end    
    gaussiannodes = filter(i -> isa(spn[i],GaussianDistribution), 1:length(spn))
    for i in gaussiannodes
        @inbounds spn[i].mean = rand()
        @inbounds spn[i].variance = 1.0
    end
end

""" 
Verifies if algorithm has converged.

Convergence is defined as absolute difference of score. 
Requires at least 2 steps of optimization.
"""
converged(learner::EMParamLearner) = learner.steps < 2 ? false : abs(learner.score - learner.prevscore) < learner.tolerance

# TODO: take filename os CSV File object as input and iterate over file to decreae memory footprint
"""
Improvement step for learning weights using the Expectation Maximization algorithm. Returns improvement in negative loglikelihood.

spn[i].weights[j] = spn[i].weights[k] * backpropagate(spn)[i]/sum(spn[i].weights[k] * backpropagate(spn)[i] for k=1:length(spn[i].children))

## Arguments

- `learner`: EMParamLearner struct
- `data`: Data Matrix
- `learningrate`: learning intertia, used to avoid forgetting in minibatch learning [default: 1.0 correspond to no inertia, i.e., previous weights are discarded after update]
- `smoothing`: weight smoothing factor (= pseudo expected count) [default: 0.1]
- `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(learner::EMParamLearner, Data::AbstractMatrix, learningrate::Float64 = 1.0, smoothing::Float64 = 0.1, minimumvariance::Float64 = learner.minimumvariance)
    
    # regularization constant for logweights (to avoid degenerate cases)
    τ = 1e-2
    
    numrows, numcols = size(Data)

    spn = learner.spn
    cache = learner.cache
    #cache = deepcopy(spn) # updates

    # # @assert numcols == spn._numvars "Number of columns should match number of variables in network."
    # m, n = size(spn._weights)
    # weights = nonzeros(spn._weights)
    # childrens = rowvals(spn._weights)
    # # oldweights = fill(τ, length(weights))
    # newweights = fill(τ, length(weights))
    # maxweights = similar(newweights)
    score = 0.0 # data loglikelihood
    sumnodes = filter(i -> isa(spn[i], SumNode), 1:length(spn))
    for i in sumnodes        
        @inbounds cache[i].weights .*= smoothing
    end
    # gaussiannodes = filter(i -> isa(spn[i],GaussianDistribution), 1:length(spn))
    # if length(gaussiannodes) > 0
    #     means = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
    #     squares = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
    #     denon = Dict{Integer,Float64}(i => 0.0 for i in gaussiannodes)
    # end
    diff = zeros(Float64,length(spn))
    values = similar(diff)
    # TODO parallalize this to exploit multicore
    for t = 1:numrows
        datum = view(Data,t,:)
        lv = logpdf!(values,spn,datum) # propagate input Data[i,:]
        @assert isfinite(lv) "logvalue of datum $t is not finite: $lv"
        score += lv
        backpropagate!(diff,spn,values) # backpropagate derivatives
        for i in sumnodes
            for (k,j) in enumerate(spn[i].children)
                # @assert isfinite(diff[i]) "derivative of node $i is not finite: $(diff[i])"
                # @assert !isnan(values[j]) "value of node $j is NaN: $(values[j])"
                if isfinite(values[j])
                    @inbounds δ = spn[i].weights[k]*diff[i]*exp(values[j]-lv) # improvement
                    @assert isfinite(δ) "improvement to weight ($i,$j):$(spn[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
                    @inbounds cache[i].weights[k] += δ
                end
            end
    #         # for j in children(spn,i) #spn._backward[i]
    #         #     oldweights[j,i] += spn._weights[j,i]*diff[i]*exp(values[j]-lv)
    #         # end
    #         for k in nzrange(spn._weights,i)
    #             j = childrens[k]
    #             #oldweights[k] += weights[k]*diff[i]*exp(values[j]-lv)
    #             Δ = log(weights[k]) + log(diff[i]) + values[j] - lv
    #             if t == 1
    #                 maxweights[k] = Δ
    #                 newweights[k] = 1.0
    #             else
    #                 if Δ > maxweights[k]
    #                     newweights[k] = exp(log(newweights[k])+maxweights[k]-Δ)+1.0
    #                     maxweights[k] = Δ
    #                 elseif isfinite(Δ) && isfinite(maxweights[k])
    #                     newweights[k] += exp(Δ-maxweights[k])
    #                 end
    #                 @assert isfinite(newweights[k]) "Infinite weight: $(newweights[k])"
    #             end
    #         end
        end
    #     for i in gaussiannodes
    #         α = diff[i]*exp(values[i]-lv)
    #         denon[i] += α
    #         means[i] += α*datum[spn[i].scope]
    #         squares[i] += α*datum[spn[i].scope]^2
    #     end
    end
    # # add regularizer to avoid degenerate distributions
    # newweights =  log.(newweights) .+ maxweights
    for i in sumnodes
        @inbounds cache[i].weights .*= learningrate/sum(cache[i].weights) # normalize weights
        @inbounds spn[i].weights .*= 1.0-learningrate # apply update with inertia strenght given by learning rate
        @assert sum(spn[i].weights) ≈ 1.0-learningrate "Unnormalized weight vector at node $i: $(sum(spn[i].weights)) | $(spn[i].weights)"
        @inbounds spn[i].weights .+= cache[i].weights
        @assert sum(spn[i].weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(spn[i].weights)) | $(spn[i].weights) | $(cache[i].weights)"
        # for (k,j) in enumerate(spn[i].children)
        #     # spn[i].weights .*= cache[i].weights
            # Z = sum(spn[i].weights)
            # spn[i].weights ./= Z
            # normexp!(spn[i].weights, prev[i].weights, τ) # if weights are in log
        # end
        #     chval = nzrange(spn._weights,i)
    #     normexp!(view(newweights,chval), view(weights,chval), τ)
    #     if !isfinite(sum(weights[chval])) # some numerical problem occurred, set weights to uniform
    #         weights[chval] .= 1/length(chval)
    #     end
    #     # @assert sum(weights[chval]) ≈ 1.0 "Unormalized weight vector: $(sum(weights[chval])) | $(weights[chval])"
    #     # add regularizer to avoid degenerate distributions
    #     # Z = sum(oldweights[:,i]) + τ*length(children(spn,i))
    #     # for j in  children(spn,i) #spn._backward[i]
    #     #    spn._weights[j,i] = (oldweights[j,i]+τ)/Z
    #     #    # if isnan(Z)
    #     #    #    @warn "Not a number for weight $i -> $j"
    #     #    # end
    #     # end
    end
    #println(spn)

    # for i in gaussiannodes
    #     spn[i].mean = means[i]/denon[i]
    #     spn[i].variance = squares[i]/denon[i] - (spn[i].mean)^2
    #     if spn[i].variance < minimumvariance
    #         spn[i].variance = minimumvariance
    #     end
    # end
    learner.steps += 1
    learner.prevscore = learner.score
    learner.score = -score/numrows
    return learner.prevscore-learner.score
end

