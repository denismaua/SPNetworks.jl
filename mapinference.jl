# Runs MAP Inference algorithms
using SumProductNetworks
import SumProductNetworks.MAP: maxproduct!, spn2bn

# spn_filename = ARGS[1]
spn_filename = "/Users/denis/code/SumProductNetworks/assets/example.pyspn.spn"

# Load SPN from spn file
spn = SumProductNetwork(spn_filename; offset = 1)
println(spn)
nvars = length(scope(spn))

# Load evidence and query variables
# TODO
x = ones(Float64, nvars)
query = Set(scope(spn))

# Run max-product 
mptime = @elapsed maxproduct!(x, spn, query)
println("MaxProduct: $(spn(x)) [$(mptime)s]")
