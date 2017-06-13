##########################################################################################
# Basic information of an Underground object
#
# An Underground object contains the basic information of a rail transport
# system, which is a collection of networks with common nodes. Nodes, here
# called 'locations' can be of three types, 'outer station', 'inner station',
# or a 'link'.
#
# A 'outer station' just symbolizes a station when a passenger is entering
# or leaving the system, while an 'inner station' represents the same
# physical location when a passenger is moving inside the system. In the logical
# network that represents our system, each outer station is linked only to its
# corresponding inner station.
#
# A 'link' is a ordered pair of two inner stations that represents a physical
# train track in the actual system. Because it is an ordered pair, the link
# is directed, and typically we expected each physical link to go in the two
# possible directions.
#
# The number of physical stations in the physical system is given by 
# 'num_stations', which is the same as 'num_outer_stations', which is the same
# as the nunber of inner stations. The number of (directed) links is given
# by 'num_dirlinks'. Hence, the number of locations is 
# 2 * num_stations + num_dirlinks.
#
# The 'locations' matrix is a num_locations x 3 matrix, where the third column
# indicates the type of location: -1 for outer stations, 0 for inner stations,
# and positive numbers for links. Each positive number indicates the line the
# link belongs to. The name of line code 'k' is given by 'line_names[k]'.
# If location in row 'i' is a directed link, locations[i, 1:2] is the ordered
# pair of inner stations corresponding to that link. Otherwise, locations[i, 1]
# is the code of the outer/inner station and locations[i, 2] = 0.
#
# Matrix 'station_pos' is a num_stations x 2 where column 1 is the latitude and
# column 2 is the  longitute of each station.
#
# It is always the case that locations from 1 to num_stations are outer stations,
# and locations num_stations + 1 to 2 * num_stations are inner stations. Moreover,
# if 'i' is an outer station, i + num_stations is the corresponding outer station.
# To find the name of an outer station i, just use station_names[i]. Similarly,
# the name of outer station i is station_names[i - num_stations].
#
# Matrix 'only_connect' is a num_locations x num_locations directed matrix. Its 
# structure reflects the physical structure. Namely, 
#
# 1. Each inner station is connected to the corresponding outer station in both 
#    directions, and outer stations have no other connections. 
#
# 2. Each inner station i is connected to any link j such that
#    locations[j, 1] == i, so only_connect[i, j] == 1. It is also the case that
#    only_connect[j, i] == 0, since this is contrary to the flow of link j.
#    On the other hand, if locations[j, 2] == i, then only_connect[j, i] == 1.
#
# 3. Moreover, each link 'j' is connected to link 'k' (that is, only_connect[j, k] == 1)
#    only if locations[j, 2] == locations[k, 1] and locations[j, 3] == locations[k, 3].
#    Changes of line are only possible by first passing through a common inner station.
#    That is if inner station 'i' is such that locations[j, 2] == locations[k, 1] == i,
#    but locations[j, 3] != locations[k, 3], then only_connect[j, k] == 0 but following
#    Item 2 above, only_connect[j, i] == 1 and only_connect[i, k] == 1.
# 
# An Underground object also contains the adjancency list representation of 
# only_connect. 'adj[i]' is a vector of integers with the codes of all locations 'j'
# such that only_connect[i, j] == 1. The length of each adj[i] is also stored in
# num_adj[i].
#
# Field 'baseline_time' indicates which real time corresponds to time 1 in this
# object. So if baseline_time is 300, then t = 1 corresponds to time 301 in real
# time, 301 minutes from midnight, or 5:01am. Field 'T' corresponds to possible
# number of time units in a day.
#
# Matrix 'allowances' is a num_stations x num_stations matrix where allowances[i, j]
# contains a vector of all possible locations in which we give a non-zero prior
# probability of being in a trajectory from outer station i to outstation j. This is
# used to dramatically reduce computation as length(allowances[i, j]) is typically
# much less than num_locations.
#
# Matrix 'paths' is a num_stations x num_stations matrix such that paths[i, j]
# is the minimum length path from outer station i to outer station j. The length of
# paths[i, j] is stored in 'path_size[i, j]'. Matrix 'mintime_taken' is a
# num_locations x num_locations matrix such that mintime_taken[i, j] gives the
# minimum amount of time, in time units (typically minutes), that it takes to go
# from location i to location j. This is used to further time-dependent zero
# prior probability of reaching a particular location from a particular location
# and time given as origin.
#
# Vector 'stay_prob' has length num_locations, and stay_prob[i] is a number in
#  [0, 1] proportional to the probability of a passenger staying at the current 
# location i even if a train is available to allow the passenger to proceed.
# The parameterization of stay_prob is given by an object of type UndergroundParams and 
# set by the function 'build_navigation_odds'.
#
# Field MIN_PROB is a number that corresponds to the probability of a passenger
# proceeding even if no train appears to allow for that to happen.
#
# Matrix 'odds' is a num_positions x num_stations matrix so that odds[i, j] is
# a vector with entries corresponding to adj[i]: here, odds[i, j][k] is a
# positive parameter indicating how favorable is it to follow a path from i to k
# in a journey that should end at outer station j. The parameterization of
# odds is given by an object of type UndergroundParams and set by the function
# 'build_navigation_odds'.

type Underground

  station_names::Array{String, 1}    
  line_names::Array{String, 1}     
  locations::Array{Int, 2}        
  station_pos::Matrix{Float64}
  only_connect::Array{Bool, 2}         

  num_stations::Int
  num_outer_stations::Int
  num_all_stations::Int
  num_locations::Int
  num_lines::Int
  num_dirlinks::Int
  num_adj::Array{Int, 1}

  baseline_time::Int
  T::Int

  allowances::Array{Array{Int16}, 2}
  paths::Array{Array{Int16}, 2}
  path_size::Array{Int16, 2}
  mintime_taken::Array{Int16, 2}
  adj::Array{Array{Int64}, 1}
  
  w::Array{Float64, 1} # Deprecated
  stay_prob::Vector{Float64}
  MIN_PROB::Float64
  odds::Array{Array{Float16}, 2}

end

const U_OUTER_STATION = -1;
const U_INNER_STATION = 0;

type UndergroundParams
  stay_prob_param::Float64
  odds_in_best_path::Float64
  odds_dirlink_not_best::Float64
  odds_station_not_best::Float64
end

type PosteriorTrajectories

  q_S_bivars::Vector{Array{Float64, 3}}
  allows::Vector{Array{Int16, 1}} 

end

type Netmis
  stations::DataFrame
  station_codes::Vector{Any}
  station_names::Vector{String}
  match_station::Dict
  lines::DataFrame
  line_codes::Vector{Any}
  line_names::Vector{String}
  match_lines::Dict
end

const NETMIS_LOAD_LOW = -1;
const NETMIS_LOAD_MEDIUM = -2;
const NETMIS_LOAD_HIGH = -3;
const NETMIS_LOAD_NA = -4;
