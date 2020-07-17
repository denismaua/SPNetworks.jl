# Test SPN to MILP translation

@testset "SPN2MILP" begin
    # SPN with dichotomic sum nodes (<= 2 children)
    SPN = SumProductNetwork(
        [
            SumNode([3,2],[0.2,0.8]),               # 1
            SumNode([4,5],[0.625,0.375]),           # 2
            ProductNode([6,8]),                     # 3
            ProductNode([6,9]),                     # 4
            ProductNode([7,9]),                     # 5
            SumNode([10,11],[0.6,0.4]),             # 6
            SumNode([10,11],[0.1,0.9]),             # 7
            SumNode([12,13],[0.3,0.7]),             # 8
            SumNode([12,13],[0.8,0.2]),             # 9
            IndicatorFunction(1,1),                 # 10
            IndicatorFunction(1,2),                 # 11
            IndicatorFunction(2,1),                 # 12
            IndicatorFunction(2,2),                 # 13
        ]
    )

    buckets = spn2milp(SPN)
    for (i,b) in enumerate(buckets)
        println(i, " ", b)
    end

end