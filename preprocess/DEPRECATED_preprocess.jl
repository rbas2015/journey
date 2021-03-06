using DataFrames

# Process of NetMIS data

PATH_NETMIS = PATH_PROJECT * "data_structure/";
NETMIS_DATE = "08-Nov-2013";

netmis_data = readtable(PATH_NETMIS * NETMIS_DATE * ".csv", header = true);
netmis = load_netmis(PATH_NETMIS);

# Time arrived  : netmis_data[t_choice, :TIMESTAMP]
# Time departed : netmis_data[t_choice, :ACTUAL_DEPARTURE_TIME]
# Station       : netmis_data[t_choice, :SUTOR_CODE]
# Line          : netmis_data[t_choice, :LINE_ID]
# Direction     : netmis_data[t_choice, :DIRECTION_CODE]
# t_choice = find(netmis_data[:, :TRAIN_IDENTIFICATION] .== 1049112);
# print(netmis_data[t_choice, [:TIMESTAMP, :ACTUAL_DEPARTURE_TIME, :SUTOR_CODE, :LINE_ID, :DIRECTION_CODE]])
# print_netmist_train(1049112, netmis_data, netmis, U)
# print_netmist_train(2003366, netmis_data, netmis, U)
# print_netmist_train(1046798, netmis_data, netmis, U)
# print_netmist_train(1046408, netmis_data, netmis, U)

occ = netmis_occupancy(netmis_data, netmis, U, false);
occ_int = occ + 1 - 1;
writecsv("/Users/ricardo/temp/occ.csv", occ_int)

function print_netmist_train(train_id::Int, netmis_data::DataFrame, netmis::Netmis, U::Underground)

  t_choice = find(netmis_data[:, :TRAIN_IDENTIFICATION] .== train_id);

  for i = t_choice
    
    if isna(netmis_data[i, :TIMESTAMP])  + isna(netmis_data[i, :ACTUAL_DEPARTURE_TIME]) +
       isna(netmis_data[i, :SUTOR_CODE]) + isna(netmis_data[i, :LINE_ID]) > 0
       continue
    end
 
    println(i)
    station = get(netmis.match_station, netmis_data[i, :SUTOR_CODE], 0);
    if station == 0 
      station_name = "UNKNOWN";
    else
      station_name = U.station_names[station];
    end

    line = get(netmis.match_lines, netmis_data[i, :LINE_ID], 0);
    if line == 0
      line_name = "UNKNOWN";
    else
      line_name = U.line_names[line];
    end

    direction = netmis_data[i, :DIRECTION_CODE];

    println("At ", station_name, ", line ", line_name, ", direction ", direction,
            ", [" * netmis_data[i, :TIMESTAMP] * "; " * netmis_data[i, :ACTUAL_DEPARTURE_TIME] * "]")

  end

end

# This goes through a NetMIS data and counts positions of trains at different locations and
# times

function netmis_occupancy(netmis_data::DataFrame, netmis::Netmis, U::Underground, verbose::Bool = false)

  # Collect basic information. Sort by time, then by train id

  n = nrow(netmis_data);
  sort_train = sortperm(netmis_data[:, :TIMESTAMP]);
  netmis_data = netmis_data[sort_train, :];
  sort_train = sortperm(netmis_data[:, :TRAIN_IDENTIFICATION]);
  netmis_data = netmis_data[sort_train, :];
  presence_grid = zeros(Bool, U.num_locations, U.T);

  sel_trains = find(isna(netmis_data[:, :TIMESTAMP])  + isna(netmis_data[:, :ACTUAL_DEPARTURE_TIME]) +
                    isna(netmis_data[:, :SUTOR_CODE]) + isna(netmis_data[:, :LINE_ID]) .== 0);
  
  adj_matrix = get_adj_matrix(U);
  line_idx = Vector{Vector{Int}}(U.num_lines);
  for i = 1:U.num_lines
    line_idx[i] = find(U.locations[:, 3] .== i);
  end

  CIRCLE_CODE = find(U.line_names .== "UG_circle")[1];
  HC_CODE = find(U.line_names .== "UG_hammersmith_city")[1];
  is_in_Hammersmith_Circle = zeros(Bool, U.num_stations);
  is_in_Hammersmith_Circle[find(sum(adj_matrix[HC_CODE], 2))] = true;  

  # Initialize

  old_train = -1;
  old_station = -1;
  old_line = -1;
  old_time_departed = -1;
  counts = 0;

  # Iterate

  #sel_trains2 = intersect(sel_trains, find(netmis_data[:, :TRAIN_IDENTIFICATION] .== 2003366));
  #sel_trains2 = intersect(sel_trains, find(netmis_data[:, :TRAIN_IDENTIFICATION] .== 1051607));
  #sel_trains2 = intersect(sel_trains, find(netmis_data[:, :TRAIN_IDENTIFICATION] .== 1046408));
  #count_all = 0;
  i_count = 0;

  for i = sel_trains2
    
    i_count += 1;
    #count_all += 1; if mod(count_all, 1000) == 0 println("Iteration $count_all"); end

    # Check on valid location

    station = get(netmis.match_station, netmis_data[i, :SUTOR_CODE], 0);
    if station == 0 old_train = -1; continue; end

    if netmis_data[i, :LINE_ID] == 13 
      # Bizarre bug in NetMIS? Apparently there are stations labeled as H & C even though
      # they are not, like Gloucester Road
      if is_in_Hammersmith_Circle[station]
        line = HC_CODE;
      else
        line = CIRCLE_CODE;
      end
    else
      line = get(netmis.match_lines, netmis_data[i, :LINE_ID], 0);
    end
    if line == 0 old_train = -1; continue; end

    # Get times and set presence at inner station "U.num_stations + station"

    time_arrived = string_to_time(netmis_data[i, :TIMESTAMP]) - U.baseline_time;
    time_departed = string_to_time(netmis_data[i, :ACTUAL_DEPARTURE_TIME]) - U.baseline_time;
    if time_arrived <= 0 || time_departed <= 0 || time_arrived > U.T || time_departed > U.T 
      old_train = -1; 
      continue
    end
    presence_grid[U.num_stations + station, time_arrived:time_departed] = true;
    
    if station == 98 println(i_count); end

    # Now detect whether train was at previous station and link should be considered

    current_train = netmis_data[i, :TRAIN_IDENTIFICATION];
    new_position = false;
    if current_train != old_train || old_line != line || old_train <= 0 || old_line <= 0 
      new_position = true;
    elseif !adj_matrix[line][old_station, station]
      new_position = true;
    elseif time_departed - old_time_departed > 10 # Don't accept very long stays within a link
      new_position = true;
    end

    if !new_position
      link_idx = find((U.locations[line_idx[line], 1] .== old_station) .* (U.locations[line_idx[line], 2] .== station))[1] + 
                 line_idx[line][1] - 1;
      presence_grid[link_idx, old_time_departed:time_departed] = true;
      counts += 1;
      if verbose
        println("      Tracking train ", current_train, " at ", U.station_names[station], " t$time_arrived")
      end
    elseif verbose
      println()
      println("Tracking train ", current_train, " at ", U.station_names[station], " t$time_arrived (after $counts iterations)")
      counts = 1;
    end

    # Update

    old_train = current_train;
    old_station = station;
    old_line = line;
    old_time_departed = time_departed;

  end

  return(presence_grid)

end

function string_to_time(input::String)
  num = split(input, ":");
  min = parse(Int, num[1]) * 60 + parse(Int, num[2]);
  return(min)
end

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
