
using DataFrames
using StringDistances

function load_underground(PATH_PROJECT::String, u_params::UndergroundParams)

  PATH_STRUCTURE = PATH_PROJECT * "data_structure/";

  ### Basic information
  
  println("Loading basic information...");

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

  ### Paths and constraints

  println("Loading location allowances...");

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
    allowances[i, j] = [new_allow; j];
  end
  close(a)

  println("Loading path information...");

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
  end
  for i = 1:num_stations
    paths[i, i] = Array{Int16}(0);
  end
  close(a)

  path_size = Array{Int16}(readtable(PATH_STRUCTURE * "dists.csv", header = false));
  mintime_taken = path_size; # TODO: replace this with actual minimum times

  ### Adjacency matrix building

  adj = Array{Array{Int}}(num_locations);
  num_adj = Array{Int}(num_locations);
  for i = 1:num_locations
    adj[i] = find(only_connect[i, :]);
    num_adj[i] = length(adj[i]);
  end
  
  ### User navigation parameters (deprecated)

  # - w[1]: coefficient corresponding to intercept
  # - w[2]: coefficient corresponding to difference in hops between the shortest path from 
  #         origin to destination that passes through h, and the shortest path from origin
  #         to destination in general
  # - w[3]: a binary indicator of whether target location is a station and not in a shortest
  #         path from origin to destination

  w = randn(3); 

  # Probability of continuing even if no train is recorded there

  MIN_PROB = 1.e-10;

  # Length of day

  baseline_time = 300;
  T = 1200;

  # Location and distance of stations

  station_pos = Matrix{Float64}(readtable(PATH_STRUCTURE * "geo_locations.csv", header = false))[:, 3:4];
  crowfly_dist = latlong_dist(station_pos[:, 1], station_pos[:, 2]);

  # Navigation oddscor

  println("Calculating route odds...");
  odds, stay_prob = build_navigation_odds(u_params, num_locations, num_stations, path_size, adj, locations);

  ### Build and return object
  
  U = Underground(station_names, line_names, locations, station_pos, only_connect,
                  num_stations, num_outer_stations, num_all_stations, num_locations,
                  num_lines, num_dirlinks, num_adj, baseline_time, T,
                  allowances, paths, path_size, mintime_taken, adj,
                  w, stay_prob, MIN_PROB, odds);
  return(U);

end

function load_netmis(PATH_NETMIS::String)

  stations = readtable(PATH_NETMIS * "netmis_stations.csv", header = true);
  lines = readtable(PATH_NETMIS * "netmis_lines.csv", header = true);

  station_codes = Vector{Any}(stations[:, :NetMIS_Sutor_Code]);
  station_names = Vector{String}(stations[:, :Station_Name]);
  match_station = netmis_namematch(station_codes, station_names, U.station_names, false);

  line_codes = Vector{Any}(lines[:, :LINE_ID]);
  line_names = Vector{String}(lines[:, :LINE_NAME]);
  match_lines = netmis_namematch(line_codes, line_names, U.line_names, false);

  ### Build and return object
  
  netmis = Netmis(stations, station_codes, station_names, match_station,
                  lines, line_codes, line_names, match_lines);
  return(netmis);  

end

function netmis_namematch(netmis_codes::Vector{Any}, netmis_names::Vector{String}, original_names::Vector{String}, verbose::Bool = false)

  n1, n2 = length(netmis_names), length(original_names);
  dists = zeros(Float64, n2);
  n_match = zeros(Int64, n1);

  for i = 1:n1

    target = uppercase(netmis_names[i]);

    if contains(target, "SIDING") || contains(target, "REVERSING BERTH") || 
       contains(target, "RECEPTION ROADS") || target == "SSL"
      continue
    end
    if target == "H&C"      
      n_match[i] = find(original_names .== "UG_hammersmith_city")[1]
      continue
    end

    for j = 1:n2
      dists[j] = compare(Levenshtein(), target, uppercase(original_names[j]));
    end
    n_match[i] = findmax(dists)[2];

  end

  if verbose
     for i = 1:n1 
       if n_match[i] > 0
         println(i, ": ", netmis_names[i], " --> ", original_names[n_match[i]]);
       else
         println(i, ": ", netmis_names[i], " --> UNMATCHED")
       end
     end
     for i = 1:n2 
       if sum(n_match .== i) == 0 
         println("This object is not in NetMIS: ", original_names[i]); 
       end
     end
  end

  n_dict = Dict(netmis_codes[i] => n_match[i] for i = 1:n1);
  return(n_dict)

end

