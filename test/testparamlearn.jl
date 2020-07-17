# Test Parameter Learning routines

@testset "SPN parameter learning" begin

println()
@info "## Learning Discrete SPNs"

# First create SPN to generate data from
A = Vector([
    SumNode(),     # 1
    ProductNode(), # 2
    ProductNode(), # 3
    ProductNode(), # 4
    CategoricalDistribution(1, [0.6,0.4]), # 5
    CategoricalDistribution(1, [0.1,0.9]), # 6
    CategoricalDistribution(2, [0.3,0.7]), # 7
    CategoricalDistribution(2, [0.8,0.2])  # 8
])
B = [
    [2, 3, 4], # 1 Sum P(A, B) 0.2,0.5,0.3
    [5, 7],    # 2 Prod P(A, B)=P(A)*P(B)
    [5, 8],    # 3 Prod P(A, B)=P(A)*P(B)
    [6, 8],    # 4 Prod P(A, B)=P(A)*P(B)
]
C = sparse([2,   3,     4],   
           [1,   1,     1],
           [0.2, 0.5, 0.3])
    
@info " Generating data set..."
SPN = SumProductNetwork{Int64}(A, B, C)
N = 10000
data = rand(SPN,N)
    
    
# Now create SPN with indicator distributions at leaves
A = Vector([
    SumNode(),     # 1
    ProductNode(), # 2
    ProductNode(), # 3
    ProductNode(), # 4
    SumNode(),     # 5
    SumNode(),     # 6
    SumNode(),     # 7
    SumNode(),     # 8
    CategoricalDistribution(1, [1.0,0.0]), # 9 X1 = 0
    CategoricalDistribution(1, [0.0,1.0]), # 10 X1 = 1
    CategoricalDistribution(2, [1.0,0.0]), # 11 X2 = 0
    CategoricalDistribution(2, [0.0,1.0])  # 12 X2 = 1
])
B = [
    [2, 3, 4], # 1 Sum P(A, B) 
    [5, 6],    # 2 Prod P(A, B)=P(A)*P(B)
    [6, 7],    # 3 Prod P(A, B)=P(A)*P(B)
    [7, 8],    # 4 Prod P(A, B)=P(A)*P(B)
    [11, 12],  # 5 Sum P(B)
    [9, 10],   # 6 Sum P(A)
    [11, 12],  # 7 Sum P(B)
    [9, 10]    # 8 Sum P(A)
]
C = sparse([2,   3,     4,  11, 12,  9, 10, 11, 12,  9, 10],   
           [1,   1,     1,   5,  5,  6,  6,  7,  7,  8,  8],
           [0.3, 0.3, 0.4,  .5, .5, .5, .5, .5, .5, .5, .5])

    @info " Creating SPN structure..."
    SPN2 = SumProductNetwork{Int64}(A, B, C)
    initialize(SPN2)
    println(SPN2)

    learner = EMParamLearner()
    @info "Running Expectation Maximization until convergence..."
    while !converged(learner) && learner.steps < 1000
        SumProductNetworks.step(learner,SPN2,data)
        # mean absolute error
        error = MAE(SPN,SPN2,data)
        println("It: $(learner.steps) \t NLL: $(-learner.score) \t MAE: $error")
    end
    # Reasonable values
    @test learner.score > -14000 && learner.score < -12000
    @test MAE(SPN,SPN2,data) < 0.02

    println()
    Z = 0.0
    for a in 1:2, b in 1:2
        ref = SPN([a,b])
        est = SPN2([a,b])
        println("S($a,$b) = $ref ≈ ", est)
        Z += est
    end
    @test Z ≈ 1.0

    println()

end # END of test set

