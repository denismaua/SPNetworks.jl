import Base.Threads.@threads

@testset "Defining and evaluating Gaussian SPNs" begin
    
    @info "# Gaussian SPNs"

    @info "## Creation"
    
    GSPN = SumProductNetwork([
        SumNode([2,3,4],[4/20,9/20,7/20]),# 1 Sum P(A, B) 0.2,0.5,0.3
        ProductNode([5,7]),               # 2 Prod P(A, B)=P(A)*P(B)
        ProductNode([5,8]),               # 3 Prod P(A,B)=P(A)*P(B)
        ProductNode([6,8]),               # 4 Prod P(A,B)=P(A)*P(B)
        GaussianDistribution(1, 2, 18),   # 5 Normal(A, mean=2, var=18)
        GaussianDistribution(1, 11, 8),   # 6 Normal(A, mean=11, var=8)
        GaussianDistribution(2, 3, 10),   # 7 Normal(B, mean=3, var=10)
        GaussianDistribution(2, -4, 7)    # 8 Normal(B, mean=-4, var=7)
    ])
    
    println(GSPN)
    
    @info "## Evaluation"
    res = GSPN([11.0,-4.0])
    println("S(11,-4) = $res")
    @test res ≈ 0.008137858167261642
    @test logpdf(GSPN,[11.0,-4.0]) ≈ -4.811228258006023

    println()
    @info "## Sampling"
    
    #N = 500000
    N = 1000
    data = rand(GSPN,N)

    max = -Inf
    amax = 0
    for n=1:N
        v = logpdf(GSPN,view(data,n,:))  # GSPN(data[n,:])
        if v > max
            max = v
            amax = n
        end
    end
    a, b = 11.0, -4.0
    ref = logpdf(GSPN,[a,b]) #GSPN([a,b])
    println("max ln S($(data[amax,:])) = $max ≈ ln S($a,$b) = $ref")
    
    println()
end # END of test set
