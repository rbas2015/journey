# Use 'worspace()' to clear REPL

########## Set up environment

using DataFrames
using Plots

include("journey/data_structures.jl")
include("journey/data_process.jl")
include("journey/util.jl")
include("journey/elbo_functions.jl")
include("journey/netmis.jl")

# Base.compilecache(PATH_PROJECT * "Journey.jl"); @everywhere push!(LOAD_PATH, PATH_PROJECT)
# using Journey

########## File reading: network structure

STAY_PROB_PARAM = 0.5;
ODDS_IN_BEST_PATH = 20.;
ODDS_DIRLINK_NOT_BEST = 10.;
ODDS_STATION_NOT_BEST = 10.;
u_params = UndergroundParams(STAY_PROB_PARAM, ODDS_IN_BEST_PATH, ODDS_DIRLINK_NOT_BEST, ODDS_STATION_NOT_BEST);
U = load_underground(PATH_PROJECT, u_params);

########## File reading: Oyster data

PATH_OYSTER = "C:/temp/nicolo/";
#PATH_OYSTER = PATH_PROJECT * "/temp/";
#oyster = Array{Int16}(readtable(PATH_OYSTER * "11640.csv", header = false, separator = ','));
oyster = Array{Int16}(readtable(PATH_OYSTER * "12365.csv", header = false, separator = ','));
num_journeys = size(oyster, 1);
j_universe = 1:num_journeys;
MINIBATCH_SIZE = 1000;

########## File reading: NetMis

# Process of NetMIS data

#PATH_NETMIS = PATH_PROJECT * "data_structure/";
#NETMIS_DATE = "08-Nov-2013";
PATH_NETMIS_STRUCTURE = PATH_PROJECT * "data_structure/";
PATH_NETMIS_DATA = PATH_OYSTER * "netmis/";
NETMIS_DATE = "02-Feb-2014";

netmis_data = readtable(PATH_NETMIS * NETMIS_DATE * ".csv", header = true);
netmis = load_netmis(PATH_NETMIS_STRUCTURE);

occ, occ_load = netmis_occupancy(netmis_data, netmis, U, true);

########################################################################## Variational evaluation

########## Allocate train evolution parameters

# Parameters for prior distribution of Z:
#
#  - for each dirlink i and t > 1, p_param_Z[i, 1, t] means P(Z_{i;t} = 1 | Z_{i;t - 1} = 0)
#  - for each dirlink i and t > 1, p_param_Z[i, 2, t] means P(Z_{i;t} = 1 | Z_{i;t - 1} = 1)
#  - for each dirlink i and t = 1, p_param_Z[i, 1, 1] means P(Z_{i;1} = 1)
#  - p_param_Z[:, 2, 1] is not used
#  - if location i is a (inner or outer) station, p_param_Z[i, :, :] is not used

p_param_Z = rand(U.num_dirlinks, 2, U.T);

# Marginal probabilities of variational distribution q(Z[i, t] == 1) for location i, time t
#
#  - for each location i and t > 1, q_Z[i, 1] means q(Z_{i;t} = 1)
#  - if location is a (inner or outer) station, q(Z_{i;t} = 1) = 1.

q_Z = rand(U.num_locations, U.T); 
q_Z = ones(U.num_locations, U.T); # TODO REMOVE THIS
q_Z[1:U.num_all_stations, :] = 1; # Stations are always accessible.

######## SIMPLE DEMONSTRATION OF TRAJECTORY COMPUTATION, GIVEN Q_Z

mini_batch = rand(j_universe, MINIBATCH_SIZE);
jny = oyster[mini_batch[1], :]; # Try journey 1692917, i.e., jny = oyster[1692917, :]; Try also 1492834
origin, destination, jny_0, jny_T = jny;
print_path(U.paths[origin, destination], U)

tic();
q_S, q_S_bivar, allow = expectation_S(jny, q_Z, U); # SOLVE MISTERY: WHY IT DOESN'T STAY? Set U.stay_prob = ones(Float64, length(U.stay_prob)) * 0.99;
toc()

# Print marginal maximum a posteriori of the trajectory

map_journey = get_map_path(q_S, q_S_bivar, allow);
print_path(map_journey, U)

######## SIMPLE DEMONSTRATION OF FULL MINIBATCH ITERATION 

N = size(oyster, 1);
MINIBATCH_SIZE = 9;
mini_batch = rand(j_universe, MINIBATCH_SIZE);
jnys = oyster[mini_batch, :];
remove_i = [];
for i = 1:size(jnys, 1)
  if jnys[i, 1] == jnys[i, 2]
    remove_i = [remove_i; i];
  end
end
jnys = jnys[setdiff(1:MINIBATCH_SIZE, remove_i), :];

lambda = extract_prior_lambda(p_param_Z, U);
rho = 1.0;
q_Z = prior_marginal(p_param_Z, U);
tic();
new_lambda, new_q_Z, new_q_Z_bivar, new_q_S_info = elbo_update(N, jnys, q_Z, lambda, rho, U);
toc()

elbo_value = elbo(jnys, new_q_S_info, p_param_Z, new_q_Z, new_q_Z_bivar, U);

### Finally, a demonstration of the whole procedure

include("journey/data_structures.jl")
include("journey/data_process.jl")
include("journey/util.jl")
include("journey/elbo_functions.jl")

MINIBATCH_SIZE = 100;
forgetting_rate = 0.9;
max_iter = 2;
q_Z, q_Z_bivar = stochastic_smoothing(oyster, forgetting_rate, MINIBATCH_SIZE, max_iter, p_param_Z, U);

########