macro is_link(x, num_all_stations)
  return :($x .> $num_all_stations)
end

macro is_inner_station(x, num_outer_stations, num_all_stations)
  return :($x > $num_outer_stations && $x <= $num_all_stations)
end

macro is_outer_station(x, num_outer_stations)
  return :($x <= $num_outer_stations)
end

macro logistic(x)
  return :(1 ./ (1 + exp(-$x)))
end

function print_path(path, U::Underground)
  for i = path
    if U.locations[i, 3] == -1
       @printf("Outer Station: %s\n", U.station_names[U.locations[i, 1]]);
    elseif U.locations[i, 3] == 0
       @printf("Inner Station: %s\n", U.station_names[U.locations[i, 1]]);
    else
       @printf("Link         : %s --> %s [%s]\n", U.station_names[U.locations[i, 1]], U.station_names[U.locations[i, 2]], U.line_names[U.locations[i, 3]]);
    end
  end 
end

# Print all information from the data structure "locations" from an Underground object
#
# f = open("c:/temp/locations.csv", "w"); print_locations(U, f); close(f)

function print_locations(U::Underground, stream = STDOUT)

  local type_str, loc_name_1, loc_name_2, line_name::String;
  local sep::String = ",";
  local code_1, code_2, line_code::Int;

  println(stream, "ID", sep, "CODE_1", sep, "CODE_2", sep, "LINE_CODE", sep, "TYPE", sep, "NAME_1", sep, "NAME_2", sep, "LINE_NAME")

  for i = 1:U.num_locations

    code_1, code_2, line_code = U.locations[i, :];

    if U.locations[i, end] == U_OUTER_STATION
      type_str   = "Outer station";
      loc_name_1 = U.station_names[i];
      loc_name_2 = "";
      line_name  = "";    
    elseif U.locations[i, end] == U_INNER_STATION
      type_str   = "Inner station";
      loc_name_1 = U.station_names[i - U.num_stations];
      loc_name_2 = "";
      line_name  = "";    
    else 
      type_str   = "Link";
      loc_name_1 = U.station_names[U.locations[i, 1]];
      loc_name_2 = U.station_names[U.locations[i, 2]];
      line_name  = U.line_names[U.locations[i, 3]];
    end 

    println(stream, i, sep, code_1, sep, code_2, sep, line_code, sep, type_str, sep, loc_name_1, sep, loc_name_2, sep, line_name)

  end 

end

# For a given origin and destination, and a particular length of time, we
# find, for every time point, which locations are reachable in this journey. 
# That boils down to find at time t the locations which can be reached from
# the origin in t hops, and which can reach the destination in time_taken - t hops.

function reachable(origin::Int16, destination::Int16, time_taken::Int, U::Underground)
  R = Array{Array{Int}}(time_taken);
  allowed = zeros(Int, U.num_locations);
  allowed[U.allowances[origin, destination]] = 1;
  for t = 1:time_taken
    forward = U.mintime_taken[origin, :] .<= t;
    backward = U.mintime_taken[:, destination] .<= time_taken - t;
    R[t] = find(forward .* backward .* allowed);
  end
  return(R);
end

function reachable_idx(adjs::Vector{Int}, reachable_bool::Vector{Bool})
  n = length(adjs);
  approve = Vector{Int}(n);
  count = 0;
  for i = 1:n
    if reachable_bool[adjs[i]]
      count += 1;
      approve[count] = i;
    end
  end
  return(approve[1:count])
end

# This function generates random posterior distributions for trajectories. The goal is to
# use this to debug the ELBO updates: if the ELBO does worse after updating, then there
# is a bug.

function random_q_S(jnys::Array{Int16, 2}, U::Underground)

  num_journeys = size(jnys, 1);
  q_S_bivars = Vector{Array{Float64, 3}}(num_journeys);
  allows = Vector{Array{Int16, 1}}(num_journeys);
  ws = Vector{Array{Float64, 1}}(num_journeys);

  for n = 1:num_journeys

    origin, destination, jny_0, jny_T = jnys[n, :];
    time_taken = jny_T - jny_0 + 1;
    allow = U.allowances[origin, destination];
    num_allow = length(allow);
    allow_idx = zeros(Int, U.num_locations);
    allow_idx[allow] = 1:num_allow;
    
    shortest_path = zeros(Int, U.num_locations);
    shortest_path[U.paths[origin, destination]] = 1;

    phi_1 = U.path_size[origin, allow] + U.path_size[allow, destination] - U.path_size[origin, destination]; 
    phi_2 = (U.locations[allow, 3] .== 0) .* (1 - shortest_path[allow]); 

    w = zeros(Float64, U.num_locations);
    w[allow] = exp(-([ones(num_allow) phi_1 phi_2] * U.b)); 
    w[w .== 0] = U.MIN_PROB / 100; 
    allow_adj = copy(U.adj);
    for s = allow
      allow_adj[s] = intersect(U.adj[s], allow);
    end
    allow_adj[destination + U.num_stations] = setdiff(allow_adj[destination + U.num_stations], destination);

    od_reach = reachable(origin, destination, time_taken, U);

    q_S = zeros(Float64, U.num_locations, time_taken);
    q_S[od_reach[1], 1] = 1; 
    q_S_bivar = zeros(Float64, num_allow, num_allow, time_taken - 1);

    for t = 1:time_taken - 1

      if t == time_taken - 1
        allow_adj[destination + U.num_stations] = [allow_adj[destination + U.num_stations]; destination];
      end

      for i = od_reach[t]

        j = [allow_adj[i]; i];
        next_j = rand(length(j)); next_j /= sum(next_j);
        q_S_bivar[allow_idx[i], allow_idx[j], t] = q_S[i, t] * next_j;

      end

      q_S[allow, t + 1] = sum(q_S_bivar[:, :, t], 2);

    end

    q_S_bivars[n] = q_S_bivar;
    allows[n] = allow;
    ws[n] = w;

  end

  ### Finalize

  q_S_info = PosteriorTrajectories(q_S_bivars, allows, ws);
  return(q_S_info)

end

# This function generates a dummy posterior distributions for the enabling variables. The goal is to
# use this to debug the ELBO updates: if the ELBO does worse after updating, then there
# is a bug.

function initial_q_Z(p_param_Z::Array{Float64, 3}, U::Underground)

  q_Z = prior_marginal(p_param_Z, U);
  q_Z_bivar = Array{Float64}(U.num_locations, 4, U.T - 1);

  for t = 1:U.T - 1
    q_Z_bivar[:, 1, t] = (1 - q_Z[:, t]) .* (1 - q_Z[:, t + 1]); # q_00
    q_Z_bivar[:, 2, t] = (1 - q_Z[:, t]) .* q_Z[:, t + 1];       # q_01
    q_Z_bivar[:, 3, t] =       q_Z[:, t] .* (1 - q_Z[:, t + 1]); # q_10
    q_Z_bivar[:, 4, t] =       q_Z[:, t] .* q_Z[:, t + 1];       # q_10
  end

  return(q_Z, q_Z_bivar)

end

# Build table of probabilities for a station S_k being in the path from S_i to S_j. This
# will be renormalized based on the state that tells the current location of a particulat
# trajectory.

function build_navigation_odds(u_params::UndergroundParams,
                               num_locations, num_stations, path_size, adj, locations)

  odds = Array{Array{Float16}}(num_locations, num_stations);

  for i = 1:num_locations

    for j = 1:num_stations

      next_best_path = path_size[i, j] - 1;
      odds_ij = Array{Float16}(length(adj[i]));
      k_idx  = 0;

      for k = adj[i]

        k_idx += 1;
        k_path = path_size[k, j];

        if k_path == next_best_path # In the shortest path
          odds_ij[k_idx] = u_params.odds_in_best_path; # For instance, in_best_path = 20;
        elseif locations[k, 3] > 0 # Directed link (that is not in the shortest path)
          # Find reverse link
          reverse_k = intersect(find(locations[:, 1] .== locations[k, 2]), find(locations[:, 2] .== locations[k, 1]))[1];
          if path_size[k, j] > path_size[reverse_k, j] # Does following k make things worse?
            odds_ij[k_idx] = 1.;
          else
            odds_ij[k_idx] = u_params.odds_dirlink_not_best; # For instance, dirlink_not_best = 10;
          end
        elseif locations[k, 3] == 0 # Inner station (that is not in the shortest path)
          if path_size[locations[k, 1], j] >= path_size[i, j] # Nothing to be gained changing here
            odds_ij[k_idx] = 1.;
          else
            odds_ij[k_idx] = u_params.odds_station_not_best; # For instance, station_not_best = 10;
          end
        else # Outer station that is not shortest path, hence not destination
          odds_ij[k_idx] = 0.;
        end

      end

      odds[i, j] = odds_ij;

    end

  end

  stay_prob = ones(num_locations) * u_params.stay_prob_param;
  return(odds, stay_prob)

end

# Find the position of the elements of v1 in v2. This assumes a lot in order
# to speed things up. In particular, it assumes the order of the elements in v1
# is respected in v2, and that all elements of v1 are indeed in v2. Otherwise,
# pos2 will eventually be bigger than the size of v2 and the usual error
# will be raised.

function sorted_find(v1::Vector, v2::Vector)
  n = length(v1);
  r = zeros(Int, n);
  pos2 = 1;
  for i = 1:n
    while true
      if v1[i] == v2[pos2]
        r[i] = pos2;
        pos2 += 1;
        break
      end
      pos2 += 1;
    end
  end
  return(r)
end

# Return most probable path according to the smoothing distribution

function get_map_path(q_S::Array{Float64, 2}, q_S_bivar::Array{Float64, 3}, allow::Vector{Int16})

  d, time_taken = length(allow), size(q_S_bivar, 3) + 1;

  # Build conditional probability factors

  p_cond = Array{Float64, 3}(d, d, time_taken);
  for t = 2:time_taken
    p_cond[:, :, t] = q_S_bivar[:, :, t - 1] ./ repeat(q_S[:, t - 1], inner = (1, d));
  end

  # Forward probability propagation

  best_fwd_prob = zeros(Float64, d, time_taken);
  best_fwd_prob[:, 1] = q_S[:, 1];
  alt_best = zeros(Int, time_taken); alt_prob, alt_best[1] = findmax(q_S[:, 1]); #TODO REMOVE THIS
  for t = 2:time_taken
    p = p_cond[:, :, t] .* repeat(best_fwd_prob[:, t - 1], inner = (1, d));
    best_fwd_prob[:, t] = maximum(p, 1);
    next_prob, alt_best[t] = findmax(p_cond[alt_best[t - 1], :, t]);
    alt_prob *= next_prob;
  end
  if sum(isinf(best_fwd_prob)) + sum(isnan(best_fwd_prob)) > 0
    error("MAXPROD algorithm failed")
  end

  # Find MAP by backtracking

  best = zeros(Int, time_taken);
  best[end] = findmax(best_fwd_prob[:, end])[2];
  for t = time_taken - 1:-1:1
    best[t] = findmax(p_cond[:, best[t + 1], t + 1] .* best_fwd_prob[:, t])[2];
  end

  alt_prob2 = 1;
  for t = 2:time_taken
    alt_prob2 *= p_cond[best[t - 1], best[t], t];
  end

  # Return, with proper mapping

  return(allow[best])

end

# Sample trajectories

function simulate_trajectories(N::Int, q_S::Array{Float64, 2}, q_S_bivar::Array{Float64, 3}, allow::Vector{Int16})

  d, time_taken = length(allow), size(q_S_bivar, 3) + 1;

  trajectories = Array{Int}(N, time_taken);
  c = Categorical(q_S[:, 1]);
  trajectories[:, 1] = rand(sampler(c), N);
  traj_prob = q_S[trajectories[:, 1], 1];

  # Build conditional probability factors

  p_cond = Array{Float64, 3}(d, d, time_taken);
  for t = 2:time_taken
    p_cond[:, :, t] = q_S_bivar[:, :, t - 1] ./ repeat(q_S[:, t - 1], inner = (1, d));
  end

  # Iterate

  for n = 1:N
    for t = 2:time_taken
      c = Categorical(p_cond[trajectories[n, t - 1], :, t]);
      trajectories[n, t] = rand(sampler(c));
      traj_prob[n] *= p_cond[trajectories[n, t - 1], trajectories[n, t], t];
    end
    trajectories[n, :] = allow[trajectories[n, :]];
  end

  # Return

  return(trajectories, traj_prob)

end

# Tranform adjacency lists of an Underground object into a array of adjacency matrices

function get_adj_matrix(U::Underground)
  adj_matrix = Vector{Matrix{Bool}}(U.num_lines);
  for i = 1:U.num_lines
    adj_matrix[i] = zeros(Bool, U.num_stations, U.num_stations);
    loc_i = find(U.locations[:, 3] .== i);
    for j = loc_i
      adj_matrix[i][U.locations[j, 1], U.locations[j, 2]] = 1;
      adj_matrix[i][U.locations[j, 2], U.locations[j, 1]] = 1;
    end
  end
  return(adj_matrix)
end

# Transform a "hh:mm:ss" string into minutes.

function string_to_time(input::String)
  num = split(input, ":");
  min = parse(Int, num[1]) * 60 + parse(Int, num[2]);
  return(min)
end

# Transform a "hh:mm:ss" string into minutes.

function string_to_time2(input::String)
  pre_num = split(input, " ");
  num = split(pre_num[2], ":");
  min = parse(Int, num[1]) * 60 + parse(Int, num[2]);
  return(min)
end


# Given latitude and longitude of a set of points, return a matrix with their
# distance in meters.

function latlong_dist(latitudes::Vector{Float64}, longitudes::Vector{Float64})

  n = length(latitudes);
  D = zeros(Float64, n, n);  

  for i = 1 : n - 1

    lat1 = latitudes[i];
    lon1 = longitudes[i];

    for j = i + 1 : n    

      lat2 = latitudes[j];
      lon2 = longitudes[j];

      phi_1 = deg2rad(lat1);
      phi_2 = deg2rad(lat2);
      delta_phi = deg2rad(lat2 - lat1);
      delta_lambda = deg2rad(lon2 - lon1);

      a = sin(delta_phi / 2.)^2 + cos(phi_1) * cos(phi_2) * sin(delta_lambda / 2.)^2;
      D[i, j] = 2 * atan2(sqrt(a), sqrt(1 - a));
      D[j, i] = D[i, j];

    end

  end

  R = 6371e3;
  D *= R;
  return(D)

end
