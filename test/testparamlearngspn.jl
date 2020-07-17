# Fix EM for Gaussian SPNs... (numerically unstable)

@testset "GSPN parameter learning" begin

    println()
    @info "## Learning Gaussian SPNs"

    # First create model to generate data 
    A = Vector([
        SumNode(),     # 1
        ProductNode(), # 2
        ProductNode(), # 3
        ProductNode(), # 4
        GaussianDistribution(1, 2, 18),  # 5
        GaussianDistribution(1, 11, 8), # 6
        GaussianDistribution(2, 3, 10),  # 7
        GaussianDistribution(2, -4, 7)   # 8
    ])
    B = [
        [2, 3, 4], # 1 Sum P(A, B) 0.2,0.5,0.3
        [5, 7],    # 2 Prod P(A, B)=P(A)*P(B)
        [5, 8],    # 3 Prod P(A, B)=P(A)*P(B)
        [6, 8],    # 4 Prod P(A, B)=P(A)*P(B)
    ]
    C = sparse([2,   3,     4],   
               [1,   1,     1],
               [4/20, 9/20, 7/20])
    
    
    SPN = SumProductNetwork{Int64}(A, B, C)
    #println(GSPN)

    @info " Generating data set..."
    N = 10000
    data = rand(SPN,N)

    # Now learn parameters from data
    SPN2 = SumProductNetwork{Int64}(deepcopy(A), deepcopy(B), deepcopy(C))
    initialize(SPN2) # random parameter initialization

    @info "Running Expectation Maximization until convergence..."
    
    learner = EMParamLearner()
    learner.minimumvariance = 0.1

    while !converged(learner) && learner.steps < 1000
        SumProductNetworks.step(learner,SPN2,data)
        # mean absolute error
        error = MAE(SPN,SPN2,data)
        println("It: $(learner.steps) \t NLL: $(-learner.score) \t MAE: $error")
    end

    @test learner.score > -60000
    @test MAE(SPN,SPN2,data) < 0.3

    for node in leaves(SPN2)
        println(node)
    end

    println()

 end # End of test set
