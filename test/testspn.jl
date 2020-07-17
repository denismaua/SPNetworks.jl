# Test creation and evaluation of SPNs

@testset "Defining and evaluating discrete SPNs" begin

    @info "# Discrete SPNs"

    @info "## Creation"
    println("Simple SPN S")
    @show SPN = SumProductNetwork(
        [
            SumNode([2,3,4],[0.2,0.5,0.3]),        # 1
            ProductNode([5,7]),                    # 2
            ProductNode([5,8]),                    # 3
            ProductNode([6,8]),                    # 4
            CategoricalDistribution(1, [0.6,0.4]), # 5
            CategoricalDistribution(1, [0.1,0.9]), # 6
            CategoricalDistribution(2, [0.3,0.7]), # 7
            CategoricalDistribution(2, [0.8,0.2])  # 8
        ]
    )
    sc = scope(SPN)
    println("Scope: ", map(string,sc))
    @test length(sc) == 2
    @test 1 in sc
    @test 2 in sc
    println()

    println("SPN encoding HMM")
    @show HMM = SumProductNetwork(
        [
         SumNode([2,3],[0.3,0.7]),      # 1
         ProductNode([4,5]),            # 2
         ProductNode([6,7]),            # 3
         CategoricalDistribution(1,[0.3,0.7]), # 4 (D1)
         SumNode([8,9],[0.5,0.5]),      # 5
         SumNode([8,9],[0.2,0.8]),      # 6
         CategoricalDistribution(1,[0.8,0.2]), # 7 (D2)
         ProductNode([10,11]),          # 8
         ProductNode([12,13]),          # 9
         CategoricalDistribution(2,[0.4,0.6]), # 10 (D3)
         SumNode([14,15],[0.6,0.4]),    # 11
         SumNode([14,15],[0.4,0.6]),    # 12
         CategoricalDistribution(2,[0.25,0.75]), # 13 (D4)
         CategoricalDistribution(3,[0.9,0.1]),   # 14 (D5)
         CategoricalDistribution(3,[0.42,0.58])  # 15 (D6)
        ]
    )
    println()

    # Selective SPN
    println("Selective SPN")
    @show selSPN = SumProductNetwork(
        [
         SumNode([2,3],[0.4,0.6]),              # 1
         ProductNode([4,5,6]),                  # 2
         ProductNode([7,8,9]),                  # 3
         IndicatorFunction(1,2.0),  # 4
         #CategoricalDistribution(1,[0.0,1.0]),  # 4
         CategoricalDistribution(2,[0.3,0.7]),  # 5
         CategoricalDistribution(3,[0.4,0.6]),  # 6
         CategoricalDistribution(2,[0.8,0.2]),  # 7
         CategoricalDistribution(3,[0.9,0.1]),  # 8
         IndicatorFunction(1,1.0) # 9
         #CategoricalDistribution(1,[1.0,0.0])   # 9
        ]
    )
    println()

    @info "## Creation"
    println("SPN encoding PSDD")
    # taken from https://github.com/UCLA-StarAI/Circuit-Model-Zoo/blob/master/psdds/little_4var.psdd
    @show PSDD = SumProductNetwork(
        [
            ProductNode([2,3]),                    # 1 (10)
            SumNode([7,6,5,4],exp.([-1.6094379124341003,-1.2039728043259361,-0.916290731874155,-2.3025850929940455])),     # 2 (9)
            SumNode([11,10,9,8],exp.([-2.3025850929940455,-2.3025850929940455,-2.3025850929940455,-0.35667494393873245])), # 3 (8)
            ProductNode([14,12]), # 4
            ProductNode([14,13]), # 5
            ProductNode([15,12]), # 6
            ProductNode([15,13]), # 7
            ProductNode([18,16]), # 8
            ProductNode([18,17]), # 9
            ProductNode([19,16]), # 10
            ProductNode([19,17]), # 11
            IndicatorFunction(4, 1.0), # 12
            IndicatorFunction(4, 2.0), # 13
            IndicatorFunction(3, 1.0), # 14
            IndicatorFunction(3, 2.0), # 15
            IndicatorFunction(2, 1.0), # 16
            IndicatorFunction(2, 2.0), # 17
            IndicatorFunction(1, 1.0), # 18
            IndicatorFunction(1, 2.0)  # 19
        ]
    )
    println()

    # @info "## Evaluation"
    results = [0.3, 0.15, 0.4, 0.15]
    for a=1:2, b=1:2
        @show v = SPN([a,b])
        #println("S($a,$b) = $v")
        @test SPN([a,b]) ≈ results[2*(a-1) + b]
        @test logpdf(SPN,[a,b]) ≈ log(SPN([a,b]))
    end
    println()

    # HMM
    results = [0.11989139999999997,0.06615860000000003,0.29298060000000004,
               0.1709694,0.0708666,0.03658340000000001,0.1561014,0.08644860000000001]
    for a=1:2, b=1:2, c=1:2
        @show v = HMM([a,b,c])
        # println("HMM($a,$b,$c) = $v")
        @test v ≈ results[4*(a-1) + 2*(b-1) + c]
    end 
    println()
    # Selective
    results = [0.432,0.048,0.108,0.012,0.048,0.072,0.112,0.168]
    for a=1:2, b=1:2, c=1:2
        @show v = selSPN(Float64[a,b,c])
        # println("SEL($a,$b,$c) = $v")
        @test v ≈ results[4*(a-1) + 2*(b-1) + c]
    end
    println()
    # PSDD
    results = [0.07, 0.27999999999999997, 0.20999999999999996, 0.14, 0.010000000000000005, 0.04000000000000001, 0.029999999999999995, 0.02000000000000001, 0.010000000000000005, 0.04000000000000001, 0.029999999999999995, 0.02000000000000001, 0.010000000000000005, 0.04000000000000001, 0.029999999999999995, 0.02000000000000001]
    for a=1:2, b=1:2, c=1:2, d=1:2
        @show v = PSDD([a,b,c,d])
        # println("PSDD($a,$b,$c,$d) = $v")
        @test v ≈ results[8*(a-1) + 4*(b-1) + 2*(c-1) + d]
    end
    println()
    
    @info "## Marginalization"
    
    @test SPN([1,NaN]) ≈ 0.45
    value = SPN([1,NaN])
    println("S(A=1) = $value")
    value = SPN([NaN, 2])
    println("S(B=2) = $value")
    @test SPN([NaN,2]) ≈ 0.3
    @test logpdf(SPN,[NaN,2]) ≈ log(0.3)
    println("HMM() ≈ $(HMM([NaN,NaN,NaN]))")
    @test logpdf(HMM,[NaN,NaN,NaN]) ≈ 0.0
    println("HMM(X1=1) ≈ $(HMM([1,NaN,NaN]))")
    @test logpdf(HMM,[1,NaN,NaN]) ≈ log(0.65)
    value = selSPN([1,NaN,NaN])
    println("S2(A=1) = $value")
    @test value ≈ 0.6
    @test logpdf(selSPN,[2,NaN,NaN]) ≈ log(0.4)

    println()
    @info "## Sampling"
    
    println("$(rand(SPN)) ~ SPN")

    #N = 10000
    N = 1000
    data = rand(SPN,N)
    counts = [0 0; 0 0]
    for i = 1:N
        counts[Int(data[i,1]), Int(data[i,2])] += 1
    end
    for a in 1:2, b in 1:2
        ref = SPN([a,b])
        println("S($a,$b) = $ref ≈ ", counts[a,b]/N)
    end
    
    println("$(map(x->round(x), rand(PSDD))) ~ PSDD")
    
    println()
    

end # END of test set


