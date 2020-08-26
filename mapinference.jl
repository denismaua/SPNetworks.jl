# Runs MAP Inference algorithms
using SumProductNetworks
import SumProductNetworks: leaves, isleaf, issum, isprod, IndicatorFunction
import SumProductNetworks.MAP: maxproduct!, localsearch!, beliefpropagation! 

if length(ARGS) < 2
    println("Usage: julia --color=yes mapinference.jl spn_filename query_filename")
    exit()
end
spn_filename = ARGS[1]
q_filename = ARGS[2]

maxinstances = 2

# Which algorithms to run?
# - mp: max-product
# - ls: Local-Search
# - bp: belief-propagation
# each algorithm uses the incumbent solution of previously ran algorithms (so order matters)
# algorithms = [:mp, :ls, :bp]
algorithms = [:mp, :bp, :ls]
# algorithms = [:mp, :ls]
# algorithms = []

# collect results (value and runtime)
results = Dict{Symbol,Array{Float64}}()
for algo in algorithms
    results[algo] = Float64[]
end

println("SPN: ", spn_filename)
println("Query: ", q_filename)
println()

# Load SPN from spn file (assume 0-based indexing)
loadtime = @elapsed spn = SumProductNetwork(spn_filename; offset = 1)
println("Loaded in $(loadtime)s ", summary(spn))
nvars = length(scope(spn))
# assignment vector
x = Array{Float64}(undef, nvars)

# maxch = 0
# for node in spn
#     if !isleaf(node)
#         global maxch = max(maxch, length(node.children))
#     end
# end
# if maxch > 4
#     @warn "maximum node indegree too large: $maxch. It is highly recommend to split nodes before running this."
# end

# Load query, evidence and marginalized variables
totaltime = @elapsed open(q_filename) do io
    inst = 1
    printstyled("╮\n"; color = :red)
    while !eof(io)
        # fill!(x, NaN) # reset configuration
        printstyled("├(", inst, ")\n"; color = :red)
        # Read query variables
        fields = split(readline(io))
        header = fields[1]
        @assert header == "q"
        query = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
        # Read marginalized variables
        fields = split(readline(io))
        header = fields[1]
        @assert header == "m"
        marg = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
        for i in marg
            x[i] = NaN
        end
        # Read evidence
        fields = split(readline(io))
        evidence = Dict{Int,Int}()
        header = fields[1]
        @assert header == "e"
        for i=2:2:length(fields)
            var = parse(Int,fields[i]) + 1
            value = parse(Int,fields[i+1]) + 1
            x[var] = value
            evidence[var] = value
        end
        # @assert length(evidence) == nevid
        # println(
        #     join(
        #         setdiff(
        #             collect(1:nvars), 
        #             union(query,keys(evidence))
        #             ) .- 1,
        #         " "
        #         )
        # )
        @assert (length(query) + length(marg) + length(evidence)) == nvars
        mpvalue = 0.0
        for algo in algorithms
            printstyled("│\t"; color = :red)
            if algo == :mp
                # Run max-product 
                runtime = @elapsed maxproduct!(x, spn, query)
                printstyled("MaxProduct: "; color = :green)
            elseif algo == :ls
                # Run local search
                runtime = @elapsed localsearch!(x, spn, query, 100)
                printstyled("LocalSearch: "; color = :green)
            elseif algo == :bp
                # Run hybrid belief propagation
                runtime = @elapsed beliefpropagation!(x, spn, query; 
                    maxiterations = 5, 
                    lowerbound = true, 
                    verbose = true,
                    warmstart = true,
                    rndminit = false)
                printstyled("BeliefPropagation: "; color = :green)
                # for i = 1:30
                #     bptime = @elapsed beliefpropagation!(x, spn, query; 
                #              maxiterations = 3, 
                #              lowerbound = true, 
                #              warmstart = false,
                #              verbose = false,
                #              rndminit = true)
                #     value = spn(x)
                #     print("$value")
                #     printstyled("  [$(bptime)s]\n"; color = :light_black)
                # end    
                # runtime = @elapsed beliefpropagation!(x, spn, query; 
                # maxiterations = 5, 
                # lowerbound = true, 
                # warmstart = true,
                # rndminit = false)
                # printstyled("BeliefPropagation: "; color = :green)
            end
            value = spn(x)
            print("$value")
            printstyled("  [$(runtime)s]\n"; color = :light_black)
            # push!(results[algo], (value,runtime))
            push!(results[algo], value)
        end
        inst += 1
        # if maximum no. of instances is reached, stop
        if inst > maxinstances
            break
        end
    end
    printstyled("╯\n"; color = :red)
end
# display sumary of results
insts = length(results[algorithms[1]])
println("Total time: $(totaltime)s\nTotal instances: $insts")
cpad(s, n::Integer, p=" ") = rpad(lpad(s,div(n+length(s),2),p),n,p) # for printing centered
columns = [:it]
widths = [5]
for algo in algorithms
    push!(columns,algo)
    push!(widths,25)
end
bordercolor = :light_cyan
headercolor = :cyan
fieldcolor = :normal
printstyled('╭', join(map(w -> repeat('─',w), widths), '┬'), '╮', '\n' ; color = bordercolor)
for (col,w) in zip(columns,widths)
    printstyled("│"; color = bordercolor)
    printstyled(cpad(string(col), w); color = headercolor)
end
printstyled("│\n"; color = bordercolor)
printstyled('├', join(map(w -> repeat('─',w), widths), '┼'), '┤', '\n' ; color = bordercolor)
for it = 1:insts
    values = Any[ it ; [ results[algo][it] for algo in columns[2:end] ] ]
    for (col,width) in zip(values, widths)
        printstyled("│"; color = bordercolor)
        printstyled(lpad(col, width-1), ' '; color = fieldcolor)
    end 
    printstyled("│\n"; color = bordercolor)
end
printstyled('├', join(map(w -> repeat('─',w), widths), '┼'), '┤', '\n' ; color = bordercolor)
values = Any[ "avg" ; [ sum(results[algo])/insts for algo in columns[2:end] ] ]
for (col,width) in zip(values, widths)
    printstyled("│"; color = bordercolor)
    printstyled(lpad(col, width-1), ' '; color = headercolor)
end 
printstyled("│\n"; color = bordercolor)
printstyled('╰', join(map(w -> repeat('─',w), widths), '┴'), '╯', '\n' ; color = bordercolor)

