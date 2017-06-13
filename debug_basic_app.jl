# Use 'worspace()' to clear REPL

using DataFrames

include("journey/data_structures.jl")
include("journey/data_process.jl")
include("journey/util.jl")
include("journey/elbo_functions.jl")

##################################################################################### Data Management

########## File reading: structure

PATH_STRUCTURE = PATH_PROJECT * "data_structure/";

station_names = Array{String}(readtable(PATH_STRUCTURE * "station_names.csv", header = false)[1]);
line_names = Array{String}(readtable(PATH_STRUCTURE * "line_names.csv", header = false)[1]);
locations = Array{Int16}(readtable(PATH_STRUCTURE * "locations.csv", header = false));
only_connect = Array{Bool}(readtable(PATH_STRUCTURE * "only_connect.csv", header = false));

num_stations = length(station_names);
num_outer_stations = num_stations; # Outer only, an alias for num_stations
num_all_stations = 2 * num_stations; # Inner and outer
num_locations = size(only_connect, 1);
num_lines = length(line_names);
num_dirlinks = num_locations - 2 * num_stations;

allowances = Array{Array{Int16}}(num_stations, num_stations);
a = open(PATH_STRUCTURE * "allowances.csv");
l_count = 0;
for line = eachline(a)
  l_count += 1;
  l = split(line, ",");  
  num_l = length(l);
  new_allow = Array{Int16}(num_l - 2);
  for k = 3:num_l
    new_allow[k - 2] = parse(Int16, l[k]);
  end
  i = parse(Int, l[1]);
  j = parse(Int, l[2]);
  allowances[i, j] = new_allow;
  allowances[j, i] = new_allow;
end
close(a)

paths = Array{Array{Int16}}(num_stations, num_stations);
a = open(PATH_STRUCTURE * "paths.csv");
l_count = 0;
for line = eachline(a)
  l_count += 1;
  l = split(line, ",");  
  num_l = length(l);
  new_path = Array{Int16}(num_l - 4);
  for k = 4:(num_l - 1)
    new_path[k - 3] = parse(Int16, l[k]);
  end
  i = parse(Int, l[1]);
  j = parse(Int, l[2]);
  paths[i, j] = new_path;
  paths[j, i] = new_path;
end
for i = 1:num_stations
  paths[i, i] = Array{Int16}(0);
end
close(a)

path_size = Array{Int16}(readtable(PATH_STRUCTURE * "dists.csv", header = false));
mintime_taken = path_size; # TODO: replace this with actual minimum times

adj = Array{Array{Int}}(num_locations);
num_adj = Array{Int}(num_locations);
for i = 1:num_locations
  adj[i] = find(only_connect[i, :]);
  num_adj[i] = length(adj[i]);
end

########## Allocate user navigation parameters

# Navigation parameters:
#
# - w[1]: coefficient corresponding to intercept
# - w[2]: coefficient corresponding to difference in hops between the shortest path from 
#         origin to destination that passes through h, and the shortest path from origin
#         to destination in general
# - w[3]: a binary indicator of whether target location is a station and not in a shortest
#         path from origin to destination

w = randn(3); 

stay_prob = ones(num_locations) * 0.5; # TODO: plug-in estimates of probability of staying at each location per unit of time
log_stay_prob = log(stay_prob);

# Probability of continuing even if no train is recorded there

MIN_PROB = 1.e-10;
log_MIN_PROB = log(MIN_PROB);

########## File reading: Oyster data

#PATH_OYSTER = "C:/temp/nicolo/";
PATH_OYSTER = PATH_PROJECT * "/temp/";
oyster = Array{Int16}(readtable(PATH_OYSTER * "11640.csv", header = false, separator = ','));
num_journeys = size(oyster, 1);
T = 1500;
j_universe = 1:num_journeys;
MINIBATCH_SIZE = 1000;

########## Create derived variables

#= path_size = Array{Int}(num_locations, num_stations);
for j = 1:num_stations
  path_size[j, j] = 0;
  for i = 1:num_stations
    path_size[i, j] = length(paths[i, j]) + 1;
    path_size[i + num_stations, j] = length(paths[i, j]);
    if i == j
      path_size[i, j] = 0;
      path_size[i + num_stations, j] += 1;
    end
  end
end
for j = 1:num_stations
  for i = (2 * num_stations + 1):num_locations
    #print(i, " ",  j, "\n")
    origin = locations[i, 1]; # Tail of edge i
    path_size[i, j] = length(paths[origin, j]) - 1; # Subtract inner station, as this is an edge
    if origin == j
      path_size[i, j] += 1;
    else
      first_edge = paths[origin, j][2]; # First directed edge out of station i towards j
      if first_edge != i
        path_size[i, j] += 1; # Shortest path from i is not really via edge locations[i, :]
      end
    end
  end
end
=#

##################################################################################### Variational evaluation



# Base.compilecache(PATH_PROJECT * "Journey.jl"); @everywhere push!(LOAD_PATH, PATH_PROJECT)
# using Journey


########## Allocate train evolution parameters

# Parameters for likelihood and variational distribution of Z: 
#
#  - first entry is logit of probabiity of being zero
#  - second entry is log-odds for the consecutive time points

q_param_Z = rand(num_locations, 2, T ) .* 2 - 1; 
p_param_Z = rand(num_locations, 2, T) .* 2 - 1;

# Marginal probabilities of variational distribution: q(Z[i, t] == 1) for location i, time t

q_Z = 1 - @logistic(q_param_Z[:, 1, :]);
q_Z[1:num_all_stations, :] = 1; # Stations are always accessible.
q_Z1 = q_Z;
q_Z0 = 1 - q_Z;

## DEBUG

tic();
elbo = full_elbo(p_param_Z, q_param_Z);
toc()

## END DEBUG

###### RELEVANT DATA STRUCTURES


U = Underground(allowances, adj, mintime_taken, station_names, line_names, locations,
                only_connect, num_stations, num_outer_stations, num_all_stations,
                num_locations, num_lines, num_dirlinks);



od_reach[t]: all locations which are physically possible to be at at time t;


###### END OF DOCUMENTING BLOCK
=#

mini_batch = rand(j_universe, MINIBATCH_SIZE);

origin, destination, jny_0, jny_T = oyster[mini_batch[1], :];
allow = [allowances[origin, destination]; destination];
num_allow = length(allow);
print_path(paths[origin, destination], locations, station_names, line_names)
time_taken = Int16(jny_T - jny_0 + 1);
allow_idx = Array{Int}(zeros(num_locations));
allow_idx[allow] = 1:num_allow;

#### Build table of probability p(location_t = i | location_{t - 1} = j, origin, destination)

jny_p = Array{Float64}(num_allow, num_allow);
station_k1, station_k2, sys_k = Int16, Int16, Int16;

shortest_path = Array{Int}(zeros(num_locations));
shortest_path[paths[origin, destination]] = 1;

phi_1 = path_size[origin, allow] + path_size[allow, destination] - path_size[origin, destination]; # Difference between shortest path containing allow[i] and shortest path
phi_2 = Array{Int}(locations[allow, 3] .== 0) .* (1 - shortest_path[allow]); # Is allow[i] an inner station, and not in the shortest path?
f_table, log_f_table = zeros(Float64, num_locations), zeros(Float64, num_locations);
f_table[allow] = exp(-[ones(num_allow) phi_1 phi_2] * w); # This table will include the unnormalized, starting-and-train-allowed probabilities
log_f_table[allow] = log(f_table[allow]);

# Debug: go through every time point where the journey took place, calculate expected log energy

od_reach = reachable(origin, destination, time_taken, mintime_taken, allowances);

#= This code does not decouple X and Z. Too slow.

for t = 2:time_taken - 1

  s1_count = 0;

  for s1 = od_reach[t - 1]

    s1_count += 1;

    num_z = length(adj[s1]);
    num_config = 2^num_z; # Go through all configurations of Z parents
    z_state = zeros(Int, num_z);
    table_output = Array{Float64}(num_adj[s1] + 1, num_config);
    @printf("num config[t = %d, s1 = %d] = %d\n", t, s1, num_config)
    for i = 1:num_config      
      score = [f_table[adj[s1]] .* z_state; stay_prob[s1]];
      table_output[:, i] = score ./ sum(score);
      #advance_binary!(z_state);
      for k = 1:length(z_state)
        if z_state[k] == 0
         z_state[k] = 1;
         break;
        end
        z_state[k] = 0;
      end
    end

  end

end
=#

# This code decouples X and Z, and uses a bound on the summation.

# - alpha: forward message
# - beta: backward message

tic();

alpha = zeros(Float64, num_locations, time_taken);
alpha[od_reach[1], 1] = 1; # Initial position
alpha[destination, end] = 1; # Final position

for t = 2:time_taken - 1

  for s1 = od_reach[t - 1]

    q = q_Z1[adj[s1], t];
    score = [f_table[adj[s1]] .* q + MIN_PROB * (1 - q); stay_prob[s1]];
    score_num = [log_f_table[adj[s1]] .* q + log_MIN_PROB * (1 - q); log_stay_prob[s1]];
    alpha[[adj[s1]; s1], t] += alpha[s1, t - 1] * exp(score_num) ./ sum(score);

  end

  alpha[od_reach[t], t] /= sum(alpha[od_reach[t], t]);

end

beta = zeros(Float64, num_locations, time_taken);
beta[destination, end] = 1; # Final  position
beta[od_reach[1], 1] = 1; # Initial position

for t = time_taken:-1:3

  for s1 = od_reach[t - 1]

    q = q_Z1[adj[s1], t];
    score = [f_table[adj[s1]] .* q + MIN_PROB * (1 - q); stay_prob[s1]];
    score_num = [log_f_table[adj[s1]] .* q + log_MIN_PROB * (1 - q); log_stay_prob[s1]];
    beta[s1, t - 1] += sum(beta[[adj[s1]; s1], t] .* exp(score_num) ./ sum(score));

  end

  beta[od_reach[t], t - 1] /= sum(beta[od_reach[t], t - 1]);

end

allow_x = [destination; allow];
q_S = alpha[allow_x, :] .* beta[allow_x, :];
q_S = q_S ./ repeat(sum(q_S, 1), inner = (length(allow_x), 1));
toc()

marg_max = zeros(Int, time_taken);
prob_loc = zeros(Float64, time_taken);
for t = 1:time_taken
  f = findmax(q_S[:, t]);
  prob_loc[t],  marg_max[t] = f[1], allow_x[f[2]];
end
print_path(marg_max, locations, station_names, line_names)

















allowed_t = origin + num_stations; # First set of possible 
allowed_next = Array{Int}(zeros(num_locations));
allowed_next[allowed_t] = 1;
for t = 2:(num_t - 1)  
  for v = allowed_t
    allowed_next[adj[v]] = allow_idx[adj[v]];
  end
  allowed_t_plus = find(allowed_next);
  l = allow_idx[allowed_t_plus]; # Index among the allowed next positions
  v1_count = Int(0);
  for v1 = allowed_t
    v1_count += 1;
    prob_next = f_table[l];
    prob_next[v1_count] = 0; # Probability of staying (before normalization)
    v2_count = Int(0);    
    for v2 = allowed_t_plus
      v2_count += 1;
      if @is_link(v2, num_all_stations)
        # With link allowed
        # With link not allowed
      else
      end
    end
  end
  allowed_t = allowed_t_plus;
end












for j = 1:num_allow # Current position
  adj_j = allow_idx[adj[locations[allow[j], 1]]];  
  for i = adj_j # Next position

  end
end



q = Array{Float64}(num_allow, num_allow, num_t);





for t = 2:num_t, j = 1:num_allow, i = 1:num_allow
   jny_p
end








########## Create data structures

# Step 1: create parameters



# Step 2: Create a prototypical variational parameter for one journey

num_t = 30; # 30 time steps for this passenger
o = 137;    # Origin
d = 2;      # Destination
s = 400;    # Starting time

# Test: how much allocation is needed for a single variational parameter?

max_T = 90; # Maximum number of time points in a journey
q = Array{Any}(max_T, num_locations);
for t = 1:max_T
  for i = 1:num_locations
    q[t, i] = Array{Float32}(num_adj[i])
  end
end

# Needed: matrix of shortest paths
# 

tic()
dummy = 0;
for t = 1:num_t
  for i = 1:100#num_locations
    for j = 1:num_adj[i]
      dummy = dummy + q[t, i][j];
    end
  end
end
toc()


#############################################################################################
# DEBUG REPARAMETERIZATION CODE

include("journey/elbo_functions.jl")

p_param_Z = rand(U.num_dirlinks, 2, U.T); # Arbitrary distribution
lambda = extract_prior_lambda(p_param_Z, U); # Reparameterize as log-exponential
p_marginals, p_bivariate = reparameterize_q_Z(lambda, U); # Reconstruction of marginals by dynamic programming
q_Z_alt = prior_marginal(p_param_Z, U); # Alternative marginalization
sum(abs(q_Z_alt - p_marginals)) # Do they agree?

###

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
new_lambda, new_q_Z, new_q_Z_bivar = elbo_update(N, jnys, q_Z, lambda, rho, U);
toc()

# Check whether marginals agree on intersections

margin_0 = new_q_Z;
margin_1 = new_q_Z_bivar[:, 3, :] + new_q_Z_bivar[:, 4, :];
margin_2 = new_q_Z_bivar[:, 2, :] + new_q_Z_bivar[:, 4, :];
m = zeros(Float64, U.num_locations);
for s = 1:U.num_locations
  m[s] = maximum(abs(margin_2[s, 1:1498] - margin_1[s, 2:1499]));
end
println(maximum(m))
for s = 1:U.num_locations
  m[s] = maximum(abs(new_q_Z[s, 1:1499] - margin_1[s, :]));
end
println(maximum(m))

### Assess ELBO

include("journey/util.jl")
include("journey/elbo_functions.jl")

dummy_q_S_info = random_q_S(jnys, U);
dummy_q_Z, dummy_q_Z_bivar = initial_q_Z(p_param_Z, U);
starting_elbo_value = elbo(jnys, dummy_q_S_info, p_param_Z, dummy_q_Z, dummy_q_Z_bivar, U);

lambda = extract_prior_lambda(p_param_Z, U);
q_Z = prior_marginal(p_param_Z, U);
new_lambda, new_q_Z, new_q_Z_bivar, new_q_S_info = elbo_update(N, jnys, q_Z, lambda, 1.0, U);
updated_elbo_value = elbo(jnys, new_q_S_info, p_param_Z, new_q_Z, new_q_Z_bivar, U);
println("ELBO changed from $starting_elbo_value to $updated_elbo_value")

####################################### Stuff with RODS

####################################### Stuff with NetMIS

good_rows = find(!isna(netmis_data[:, :PASSENGER_LOAD_STATUS]));
print(netmis_data[good_rows, [:TIMESTAMP, :ACTUAL_DEPARTURE_TIME, :SUTOR_CODE, :LINE_ID, :DIRECTION_CODE, :PASSENGER_LOAD_STATUS]])

print_netmis_data(good_rows, netmis_data, netmis, U)