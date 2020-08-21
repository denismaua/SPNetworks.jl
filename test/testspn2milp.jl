# Test SPN to MILP translation
import SumProductNetworks.MAP: spn2milp
using Gurobi

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

    model = spn2milp(SPN, [6,7,2,9,1,8])
    # println(model)
    optimize(model)
     # show results
    sol = get_solution(model)
    println("Solution: $(sol)")
    # show obj value
    obj = get_objval(model)
    println("Objective: ", obj)
    @test obj ≈ 0.4
    # for (i,b) in buckets
    #         println(i, " ", b)
    # end

end