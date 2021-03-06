# Test SPN learn of mixtures of Chow-Liu trees
@testset "Chow-Liu" begin
    import SPNetworks.BayesianNetworks: BayesianTree, BTRoot, BTNode, BTLeaf, compile
    # Example from Strudel paper (PGM 2020)
    tree = BTRoot(
        4,
        [0.6, 0.4],
        [
            BTNode(
                3,
                [0.2 0.7; 
                 0.8 0.3],
                [
                    BTLeaf(
                        1,
                        [0.3 0.6; 
                         0.7 0.4]
                    ),
                    BTLeaf(
                        2,
                        [0.5 0.1; 
                         0.5 0.9]
                    )            
                ]
            )
        ]
    )
    spn = compile(tree)
    @debug spn
    @test length(spn) == 25
    @test length(scope(spn)) == 4
    @test spn([NaN,NaN,NaN,NaN]) ≈ 1.0
    # joint probability values
    values =  Dict( (1,1,1,1) => 0.6*0.2*0.3*0.5, # x1=1, x2=1, x3=1, x4=1
                    (1,1,1,2) => 0.4*0.7*0.3*0.5, # x1=1, x2=1, x3=1, x4=2
                    (1,1,2,1) => 0.6*0.8*0.6*0.1, # x1=1, x2=1, x3=2, x4=1
                    (1,1,2,2) => 0.4*0.3*0.6*0.1, # x1=1, x2=1, x3=2, x4=2
                    (1,2,1,1) => 0.6*0.2*0.3*0.5,
                    (1,2,1,2) => 0.4*0.7*0.3*0.5,
                    (1,2,2,1) => 0.6*0.8*0.6*0.9, 
                    (1,2,2,2) => 0.4*0.3*0.6*0.9, 
                    (2,1,1,1) => 0.6*0.2*0.7*0.5, # x1=2, x2=1, x3=1, x4=1
                    (2,1,1,2) => 0.4*0.7*0.7*0.5, # x1=2, x2=1, x3=1, x4=2
                    (2,1,2,1) => 0.6*0.8*0.4*0.1, # x1=2, x2=1, x3=2, x4=1
                    (2,1,2,2) => 0.4*0.3*0.4*0.1, # x1=2, x2=1, x3=2, x4=2
                    (2,2,1,1) => 0.6*0.2*0.7*0.5,
                    (2,2,1,2) => 0.4*0.7*0.7*0.5,
                    (2,2,2,1) => 0.6*0.8*0.4*0.9, 
                    (2,2,2,2) => 0.4*0.3*0.4*0.9,                     
                )
    for x1=1:2, x2=1:2, x3=1:2, x4=1:2
        @test spn(Float64[x1,x2,x3,x4]) ≈ values[x1,x2,x3,x4]
    end
end