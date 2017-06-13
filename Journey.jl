__precompile__()

module Journey

include("journey/data_structures.jl")
include("journey/data_process.jl")
include("journey/util.jl")
include("journey/elbo_functions.jl")

export full_elbo

end
