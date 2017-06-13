##########################################################################################
# Utility functions for network manipulation
#
# Functions in this file provide basic tools for network manipulation and summarization.

"""
  print_path(path::Vector{Integer}, U::Underground)

Print the names of each station in the sequence given by ``path``
according to the station names in Underground object ``U``.
"""

function print_path(path::Vector{Integer}, U::Underground)
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

"""
  reachable(origin::Integer, destination::Integer, time_taken::Integer, U::Underground)

For a given origin and destination, and a particular length of time, we
find, for every time point, which locations are reachable in this journey. 
That boils down to find at time t the locations which can be reached from
the origin in t hops, and which can reach the destination in time_taken - t hops.
"""

function reachable(origin::Integer, destination::Integer, time_taken::Integer, U::Underground)
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

function reachable_idx(adjs::Vector{Integer}, reachable_bool::Vector{Bool})
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

"""
  get_adj_matrix(U::Underground)

Tranform adjacency lists of an Underground object into a array of adjacency matrices.
"""

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
