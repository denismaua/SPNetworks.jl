# Test creation of template SPNs

@testset "Defining and evaluating discrete SPNs" begin

    @info "# Chain Rule SPN"

    @info "## Creation"
    println("SPN with single ternary variable")

    SPN = chainruleSPN([3]) 
    println(SPN)
    @test SPN([NaN]) ≈ 1.0
    println("SPN(1) = $(SPN([1.0]))")
    @test SPN([1.0]) ≈ SPN([2.0]) ≈ SPN([3.0]) ≈ 1/3
    # TODO: generate some data and learn parameters from them 
    println()
    
    @info "## Creation"
    println("SPN with 3 binary variables")

    SPN = chainruleSPN([2,2,2]) 
    println(SPN)
    #@test length(SPN) == 19
    #@test size(SPN) == 14
    #@test length(SumProductNetworks.leaves(SPN)) == 10
    #@test length(sumnodes(SPN)) == 3
    #@test length(productnodes(SPN)) == 6
    for i=1:length(SPN)
        if isa(SPN[i],LeafNode)
            println("$i.  ",SPN[i])
        else
            println("$i.  ",SPN[i])
        end
    end
    println()
    @test SPN([NaN,NaN,NaN]) ≈ 1.0
    for a=1:2,b=1:2,c=1:2
        println("SPN($a,$b,$c) = $(SPN([a,b,c]))")
        @test SPN([a,b,c]) ≈ 1/8
    end
    # TODO: generate some data and learn parameters from them 
    # XOR, _ = SumProductNetworks.load("../examples/xor.spn")
    # N = 1000
    # data = rand(XOR,N)
    # for a in 1:2, b in 1:2, c in 1:2
    #     ref = XOR([a,b,c])
    #     println("SPN($a,$b,$c) = $ref ≈ ", SPN([a,b,c]))
    # end
    println()
    # HMM, _ = SumProductNetworks.load("../examples/hmm.spn")
    # TODO: Learn parameters from real dataset, e.g. NLTCS
    # @time BIG = chainruleSPN(2*ones(Int64,18))
    # println(BIG)

end # END of test set
