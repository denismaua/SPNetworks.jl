# Compares EM variants on given network structure
using DataFrames, CSV
using Random
using SPNetworks
import SPNetworks: NLL
import SPNetworks.ParameterLearning: EMParamLearner, initialize, converged, update, backpropagate

if length(ARGS) < 3
    println("Usage: julia --color=yes em.jl spn_filename train_filename validation_filename [batchsize]")
    exit()
end
spn_filename = normpath(ARGS[1])
train_filename = normpath(ARGS[2])
valid_filename = normpath(ARGS[3])
batchsize = 100 # default
if length(ARGS) >= 3
    batchsize = parse(Int, ARGS[4])
end

spn = SumProductNetwork(spn_filename)
#SPNetworks.save(spn,normpath("$train_filename"))
nvars = length(scope(spn))
println("SPN: $spn_filename ($(summary(spn)))")
println("Training data: $train_filename")
println("Validation data: $valid_filename")
println("Maximum batchsize: $batchsize")
# Load datasets
tdata = convert(Matrix,
            DataFrame(
                CSV.File(train_filename, 
                header=collect(1:nvars) # columns names
))) .+ 1;
@show summary(tdata)
vdata = convert(Matrix,
            DataFrame(
                CSV.File(valid_filename, 
                header=collect(1:nvars) # columns names
))) .+ 1;
@show summary(vdata)
# initialize EM learner
learner = EMParamLearner(spn)
println("It: $(learner.steps) \t NLL: $(learner.score) \t held-out NLL: $(NLL(spn, vdata))")
initialize(learner) # generate random weights for each sum node
indices = shuffle!(Vector(1:size(tdata,1)))
# Running Expectation Maximization
while !converged(learner) && learner.steps < 10
    sid = rand(1:(length(indices)-batchsize))
    batch = view(tdata, indices[sid:(sid+batchsize-1)], :) # extract minibatch sample
    if learner.steps % 2 == 0          
        tnll = NLL(spn, vdata)
        update(learner, batch, 0.9^learner.steps)
        println("It: $(learner.steps) \t NLL: $(learner.score) \t held-out NLL: $tnll")
    else
        update(learner, batch, 0.9^learner.steps)
        println("It: $(learner.steps) \t NLL: $(learner.score)")
    end
end
