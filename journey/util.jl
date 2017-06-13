
##########################################################################################
# Utility functions.
#
# These are general utility functions. For more specific ones, see the files linked below.

include("util_net.jl")
include("util_infer.jl")

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
