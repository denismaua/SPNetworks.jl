# Compares SQUAREEM variants on given network structure
# Example of commaind line call: 
#     julia --threads=4 --color=yes squarem.jl assets/nltcs.spn assets/nltcs.train.csv assets/nltcs.valid.csv 
#using DataFrames, CSV
using DelimitedFiles # much faster than above
using Random
using SPNetworks
import SPNetworks: NLL
import SPNetworks.ParameterLearning: SQUAREM, initialize, converged, update

if length(ARGS) < 3
    println("Usage: julia --color=yes em.jl spn_filename train_filename validation_filename")
    exit()
end
spn_filename = normpath(ARGS[1])
train_filename = normpath(ARGS[2])
valid_filename = normpath(ARGS[3])
println("Threads: ", Threads.nthreads())

println("SPN: $spn_filename")
spn = SumProductNetwork(spn_filename)
@show summary(spn)
nvars = length(scope(spn))
println("Training data: $train_filename")
# Load datasets
@time tdata = readdlm(train_filename, ',', Float64) .+ 1;
@show summary(tdata)
println("Validation data: $valid_filename")
@time vdata = readdlm(valid_filename, ',', Float64) .+ 1;
@show summary(vdata)
# initialize EM learner
learner = SQUAREM(spn)
println("It: $(learner.steps) \t NLL: $(NLL(spn, tdata)) \t held-out NLL: $(NLL(spn, vdata))")
#Random.seed!(3)
#initialize(learner) # generate random weights for each sum node
# Running Expectation Maximization
#while !converged(learner) && 
while learner.steps < 20 # 11
    @time score, α = update(learner, tdata) 
    testnll = NLL(spn, vdata)
    trainnll = NLL(spn, tdata)
    println("It: $(learner.steps) \t NLL: $trainnll \t held-out NLL: $testnll \t α: $α")
end
