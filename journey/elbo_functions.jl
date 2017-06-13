# This function calculates marginal and bivariate distributions
# of a particular journey trajectory given the variational distribution of
# enabling variables.

function expectation_S(jny::Vector{Int16}, q_Z::Array{Float64, 2}, U::Underground)

  ###################
  # Basic information

  origin, destination, jny_0, jny_T = jny;
  time_taken = jny_T - jny_0 + 1;
  allow = U.allowances[origin, destination];
  num_allow = length(allow);
  allow_idx = zeros(Int, U.num_locations);
  allow_idx[allow] = 1:num_allow;
    
  #######################################
  # Structure information for calculation

  allow_adj = copy(U.adj);   # Remove adjacencies that are not allowed
  allow_odds = Vector{Array{Float16}}(U.num_locations); # Remove adjacencies that are not allowed
  for s = allow
    allow_adj[s] = intersect(U.adj[s], allow);
    allow_odds[s] = U.odds[s, destination][sorted_find(allow_adj[s], U.adj[s])];
  end

  # For each time point, get an array of which locations can be reached given the
  # origin and destination

  od_reach = reachable(origin, destination, time_taken, U);

  # Forward messages

  fwd = zeros(Float64, U.num_locations, time_taken);
  fwd[od_reach[1], 1] = 1; # Initial position, inner station
  gammas = Array{Array{Float64}}(U.num_locations, time_taken);

  for t = 2:time_taken

    for i = od_reach[t - 1]
      
      if i == destination continue; end

      J_i           = [allow_adj[i]; i];
      num_J_i       = length(J_i);
      q             = q_Z[J_i, t];
      w_J           = [allow_odds[i] / sum(allow_odds[i]) * (1 - U.stay_prob[i]); U.stay_prob[i]];
      gammas[i, t]  = zeros(Float64, num_J_i);

      for j = 1:num_J_i
        for j_prime = 1:num_J_i
          if j == j_prime continue; end
          gammas[i, t][j] -=        q[j] *       q[j_prime] * log(1 + w_J[j_prime] / w_J[j]) + 
                              (1 - q[j]) *       q[j_prime] * log(1 + w_J[j_prime] / U.MIN_PROB) +
                                    q[j] * (1 - q[j_prime]) * log(1 + U.MIN_PROB / w_J[j]) +
                              (1 - q[j]) * (1 - q[j_prime]) * log(2);
        end
      end
      gammas[i, t] = exp(gammas[i, t]);

      fwd[J_i, t] += fwd[i, t - 1] * gammas[i, t];

    end

    if t < time_taken fwd[destination, t] = 0; end
    fwd[allow, t] /= sum(fwd[allow, t]);

  end
  
  # Backward messages

  bwd = zeros(Float64, U.num_locations, time_taken);
  bwd[destination, end] = 1; # Final position

  for t = time_taken:-1:2

    for i = od_reach[t - 1]
      if i == destination continue; end
      j = [allow_adj[i]; i];
      bwd[i, t - 1] += sum(bwd[j, t] .* gammas[i, t]);
    end

    bwd[allow, t - 1] /= sum(bwd[allow, t - 1]);

  end

  # Univariate marginals

  q_S = fwd[allow, :] .* bwd[allow, :];
  q_S = q_S ./ repeat(sum(q_S, 1), inner = (length(allow), 1));
  
  # Bivariate marginals

  q_S_bivar = zeros(Float64, num_allow, num_allow, time_taken - 1);

  for t = 1:time_taken - 1

    #reach_idx = zeros(Bool, U.num_locations); 
    #reach_idx[od_reach[t + 1]] = true;

    for i = od_reach[t]
      if i == destination continue; end
      j = [allow_adj[i]; i];
      #a = reachable_idx(j, reach_idx);
      #j = j[a];      
      q_S_bivar[allow_idx[i], allow_idx[j], t] = fwd[i, t] * (bwd[j, t + 1] .* gammas[i, t + 1]);
      if isnan(sum(q_S_bivar[allow_idx[i], allow_idx[j], t]))
        println("NaN at ", i)
      end
    end

    q_S_bivar[:, :, t] /= sum(q_S_bivar[:, :, t]);

  end

  return(q_S, q_S_bivar, allow, fwd, bwd)

end

# Generate log-linear representation of prior of enabling variables

function extract_prior_lambda(p_param_Z::Array{Float64, 3}, U::Underground)

  lambda = Array{Float64}(U.num_dirlinks, 6, U.T);

  lambda[:, 1:2, :] = 0;
  lambda[:, 3,   :] = log(1 - p_param_Z[:, 1, :]);
  lambda[:, 4,   :] = log(p_param_Z[:, 1, :]);
  lambda[:, 5,   :] = log(1 - p_param_Z[:, 2, :]);
  lambda[:, 6,   :] = log(p_param_Z[:, 2, :]);

  lambda[:, 1,   1] = log(1 - p_param_Z[:, 1, 1]);
  lambda[:, 2,   1] = log(p_param_Z[:, 1, 1]);
  lambda[:, 3:6, 1] = 0;

  return(lambda)

end

# Provides the marginal distribution of Z according to prior. Useful for
# initialization purposes

function prior_marginal(p_param_Z::Array{Float64, 3}, U::Underground)
  q_Z = zeros(Float64, U.num_dirlinks, U.T);
  q_Z[:, 1] = p_param_Z[:, 1, 1];
  for t = 2:U.T
    q_Z[:, t] = (1 - q_Z[:, t - 1]) .* p_param_Z[:, 1, t] + 
                    (q_Z[:, t - 1]) .* p_param_Z[:, 2, t];
  end
  q_Z = [ones(Float64, U.num_all_stations, U.T); q_Z];
  return(q_Z)
end

# This function does the actual updates.

function elbo_update(N::Int, jnys::Array{Int16, 2},                        
                     q_Z::Array{Float64, 2}, old_lambda::Array{Float64, 3},
                     rho::Float64, U::Underground)

  num_journeys = size(jnys, 1);
  q_S_bivars = Vector{Array{Float64, 3}}(num_journeys);
  allows = Vector{Array{Int16, 1}}(num_journeys);
  batch_weight = N / num_journeys;

  ### Obtain evidence from trajectories

  println("   [1/3] Computing evidence...")

  for i = 1:num_journeys
    if jnys[i, 1] == jnys[i, 2]
      # Do NOT allow journeys with the same starting and ending point
      error("No journey with destination equal to origin is allowed")
    end
    dummy, q_S_bivars[i], allows[i] = expectation_S(jnys[i, :], q_Z, U);
  end

  ### Absorb prior

  lambda = extract_prior_lambda(p_param_Z, U);

  ### Absorb evidence

  println("   [2/3] Absorbing evidence...")
  allow_idx = zeros(Int, U.num_locations);
  lambda_0 = zeros(Float64, U.num_locations, U.T);
  lambda_1 = zeros(Float64, U.num_locations, U.T);

  for n = 1:num_journeys

    allow     = allows[n];
    q_S_bivar = q_S_bivars[n];
    num_allow = length(allow);
    
    origin, destination, jny_0, jny_T = jnys[n, :];
    time_taken = jny_T - jny_0 + 1;
    od_reach = reachable(origin, destination, time_taken, U);
    allow_adj = copy(U.adj); # Remove adjacencies that are not allowed    
    allow_odds = Vector{Array{Float16}}(U.num_locations); # Remove adjacencies that are not allowed
    for s = allow
      allow_adj[s] = intersect(U.adj[s], allow);
      allow_odds[s] = U.odds[s, destination][sorted_find(allow_adj[s], U.adj[s])];
    end
    allow_idx[allow] = 1:num_allow;

    for t = 2:time_taken

      for i = od_reach[t - 1]

        if i == destination continue; end

        J_i = [allow_adj[i]; i];        
        w   = [allow_odds[i] / sum(allow_odds[i]) * (1 - U.stay_prob[i]);  U.stay_prob[i]];
        j_count = 0;

        for j = J_i
          
          j_count += 1;
          q_ij = q_S_bivar[allow_idx[i], allow_idx[j], t - 1];
          j_prime_count = 0;

          for j_prime = J_i
            j_prime_count += 1;
            if j == j_prime continue; end
            lambda_0[j, t] -= q_ij * (q_Z[j_prime, t]       * log(1 + w[j_prime_count] / U.MIN_PROB) +
                                      (1 - q_Z[j_prime, t]) * log(2));
            lambda_1[j, t] -= q_ij * (q_Z[j_prime, t]       * log(1 + w[j_prime_count] / w[j_count]) +
                                      (1 - q_Z[j_prime, t]) * log(1 + U.MIN_PROB / w[j_count]));                               
          end

        end

      end

    end

  end

  lambda[:, 1, :] += batch_weight * lambda_0[U.num_all_stations + 1:end, :];
  lambda[:, 2, :] += batch_weight * lambda_1[U.num_all_stations + 1:end, :];

  lambda[:, 1, :] = (1 - rho) * old_lambda[:, 1, :] + rho * lambda[:, 1, :];
  lambda[:, 2, :] = (1 - rho) * old_lambda[:, 2, :] + rho * lambda[:, 2, :];
  
  ### Message passing for reparameterization

  println("   [3/3] Reparameterizing smoothing distribution...")
  q_Z_marginal, q_Z_bivar = reparameterize_q_Z(lambda, U);

  ### Finalize

  q_S_info = PosteriorTrajectories(q_S_bivars, allows);

  return(lambda, q_Z_marginal, q_Z_bivar, q_S_info)

end

# Transforms a set of Markov chains parameterized with log-linear
# parameters to get the univariate marginals. The outcome is
# such that q_Z[i, t] is P(Z_i;t = 1). Notice that q_Z_bivar
# is never used for ELBO updates, only when we want to compute
# the ELBO itself.

function reparameterize_q_Z(lambda::Array{Float64,3}, U::Underground)

  q_Z = zeros(Float64, U.num_locations, U.T);
  q_Z[1:U.num_all_stations, :] = 1;

  # Preliminary processing

  lambda4 = lambda[:, 3:6, :];
  lambda4[:, 1:2, :] += lambda[:, 1:2, :];
  lambda4[:, 3:4, :] += lambda[:, 1:2, :];
  theta = Array{Float64}(U.num_dirlinks, 4, U.T);
  for t = 1:U.T
     lt = lambda4[:, :, t];
     theta[:, :, t] = exp(lt - repeat(maximum(lt, 2), inner = (1, 4)));    
  end

  # Forward messages

  fwd = ones(Float64, U.num_dirlinks, 2, U.T);
  fwd[:, 1, 1] = exp(lambda[:, 1, 1]);
  fwd[:, 2, 1] = exp(lambda[:, 2, 1]);
  s = fwd[:, 1, 1] + fwd[:, 2, 1];
  fwd[:, 1, 1] ./= s;
  fwd[:, 2, 1] ./= s;

  for t = 2:U.T

    fwd[:, 1, t] = fwd[:, 1, t - 1] .* theta[:, 1, t] +
                   fwd[:, 2, t - 1] .* theta[:, 3, t];
    fwd[:, 2, t] = fwd[:, 1, t - 1] .* theta[:, 2, t] +
                   fwd[:, 2, t - 1] .* theta[:, 4, t];
    s = fwd[:, 1, t] + fwd[:, 2, t];
    fwd[:, 1, t] ./= s;
    fwd[:, 2, t] ./= s;

  end

  # Backward messages
   
  bwd = ones(Float64, U.num_dirlinks, 2, U.T);
  bwd[:, :, end] = 1;

  for t = U.T:-1:2

    bwd[:, 1, t - 1] = bwd[:, 1, t] .* theta[:, 1, t] +
                       bwd[:, 2, t] .* theta[:, 2, t];
    bwd[:, 2, t - 1] = bwd[:, 1, t] .* theta[:, 3, t] +
                       bwd[:, 2, t] .* theta[:, 4, t];
    s = bwd[:, 1, t - 1] + bwd[:, 2, t - 1];
    bwd[:, 1, t - 1] ./= s;
    bwd[:, 2, t - 1] ./= s;

  end

  # Univariate marginals

  q_Z_extra = fwd .* bwd;
  s = q_Z_extra[:, 1, :] + q_Z_extra[:, 2, :];
  q_Z[U.num_all_stations + 1:end, :] = q_Z_extra[:, 2, :] ./ s;

  problems = find(sum(s .== 0, 2));
  for i = problems
    q_Z[U.num_all_stations + i, s[i, :] .== 0] = 0.5;
  end

  # Bivariate marginals

  q_Z_bivar = Array{Float64}(U.num_locations, 4, U.T - 1);
  q_Z_bivar[1:U.num_all_stations, 1:3, :] = 0;
  q_Z_bivar[1:U.num_all_stations, 4,   :] = 1;
  dirlink_range = U.num_all_stations + 1:U.num_locations;
  q_local = Array{Float64}(U.num_dirlinks, 4);

  for t = 1:U.T - 1

    q_local[:, 1] = fwd[:, 1, t] .* bwd[:, 1, t + 1] .* theta[:, 1, t + 1]; #q_00
    q_local[:, 2] = fwd[:, 1, t] .* bwd[:, 2, t + 1] .* theta[:, 2, t + 1]; #q_01
    q_local[:, 3] = fwd[:, 2, t] .* bwd[:, 1, t + 1] .* theta[:, 3, t + 1]; #q_10
    q_local[:, 4] = fwd[:, 2, t] .* bwd[:, 2, t + 1] .* theta[:, 4, t + 1]; #q_11
    q_local ./= repeat(sum(q_local, 2), inner = (1, 4));    
    
    q_Z_bivar[dirlink_range, :, t] = q_local;

  end

  # Return

  return(q_Z, q_Z_bivar)

end

# Calculate the ELBO given data and posterior information.

function elbo(jnys::Array{Int16, 2}, q_S_info::PosteriorTrajectories,
              p_param_Z::Array{Float64, 3}, q_Z::Array{Float64, 2}, q_Z_bivar::Array{Float64, 3}, 
              U::Underground)

  value = 0.0;
  dirlink_index = U.num_all_stations + 1:U.num_locations; 

  # Expected log-prior
 
  link_q_Z2 = q_Z_bivar[dirlink_index, :, :];
  value += sum(link_q_Z2[:, 1] .* p_param_Z[:, 1, 1]);   
  for t = 2:U.T
    value += sum(link_q_Z2[:, 1, t - 1] .* log(1 - p_param_Z[:, 1, t])) + 
             sum(link_q_Z2[:, 2, t - 1] .* log(p_param_Z[:, 1, t])) +
             sum(link_q_Z2[:, 3, t - 1] .* log(1 - p_param_Z[:, 2, t])) +
             sum(link_q_Z2[:, 4, t - 1] .* log(p_param_Z[:, 2, t]));
  end

  # Entropy of distribution of enabling variables

  link_q_Z = q_Z[dirlink_index, :];
  v = link_q_Z .* log(link_q_Z) +  (1 - link_q_Z) .* log(1 - link_q_Z);
  v[link_q_Z .== 0] = 0;
  v[link_q_Z .== 1] = 0;
  value += sum(v);
  v = link_q_Z2 .* log(link_q_Z2);
  v[link_q_Z2 .== 0] = 0;
  value -= sum(v);

  # Expected log-likelihood of trajectories

  num_journeys = size(jnys, 1);
  allow_idx = zeros(Int, U.num_locations);

  for n = 1:num_journeys
    
    allow     = q_S_info.allows[n];
    q_S_bivar = q_S_info.q_S_bivars[n];
    num_allow = length(allow);
    
    origin, destination, jny_0, jny_T = jnys[n, :];
    time_taken = jny_T - jny_0 + 1;
    od_reach = reachable(origin, destination, time_taken, U);
    allow_adj = copy(U.adj); # Remove adjacencies that are not allowed    
    allow_odds = Vector{Array{Float16}}(U.num_locations); # Remove adjacencies that are not allowed
    for s = allow
      allow_adj[s] = intersect(U.adj[s], allow);
      allow_odds[s] = U.odds[s, destination][sorted_find(allow_adj[s], U.adj[s])];
    end
    allow_idx[allow] = 1:num_allow;

    for t = 2:time_taken

      for i = od_reach[t - 1]

        if i == destination continue; end

        J_i = [allow_adj[i]; i];  
        w   = [allow_odds[i] / sum(allow_odds[i]) * (1 - U.stay_prob[i]); U.stay_prob[i]];     
        j_count = 0;
        
        for j = J_i
          
          j_count += 1;
          q_ij = q_S_bivar[allow_idx[i], allow_idx[j], t - 1];
          j_prime_count = 0;

          for j_prime = J_i
            j_prime_count += 1;
            if j == j_prime continue; end
            E_z = (1 - q_Z[j, t]) * (1 - q_Z[j_prime, t]) * log(2) +
                       q_Z[j, t]  * (1 - q_Z[j_prime, t]) * log(1 + U.MIN_PROB / w[j_count]) + 
                  (1 - q_Z[j, t]) * q_Z[j_prime, t]       * log(1 + w[j_prime_count] / U.MIN_PROB) +
                       q_Z[j, t]  * q_Z[j_prime, t]       * log(1 + w[j_prime_count] / w[j_count]);
            value -= q_ij * E_z;
          end

        end

      end

    end
    

  end

  # Finalize

  return(value)

end

# MAIN FUNCTION: learn the smoothing distribution using stochastic optimization.
# The data is given in table 'journeys' which contains the columns 
# origin, destination, starting time and ending time.

function stochastic_smoothing(journeys::Array{Int16, 2}, 
                              forgetting_rate::Float64, minibatch_size::Int, max_iter::Int,
                              p_param_Z::Array{Float64, 3}, U::Underground)


  N = size(journeys, 1);
  j_universe = 1:N;
  q_Z, q_Z_bivar = initial_q_Z(p_param_Z, U);
  lambda = extract_prior_lambda(p_param_Z, U);  

  println()
  println("** JOURNEY: Smoothing distributions for system enabling variables")
  println("Doing $max_iter iterations with forgettting rate $forgetting_rate")
  println("Minibatch size is $minibatch_size")
  println()
  
  for iter = 1:max_iter

    # Select minibatch and remove journeys where origin and destination are the same

    mini_batch = rand(j_universe, minibatch_size);
    jnys = journeys[mini_batch, :];
    remove_od = [];
    for j = 1:size(jnys, 1)
      if jnys[j, 1] == jnys[j, 2]
        remove_od = [remove_od; j];
      end
    end
    jnys = jnys[setdiff(1:minibatch_size, remove_od), :];
    
    # Update and report ELBO on minibatch

    rho = 1. / (iter + 1)^forgetting_rate;
    lambda, q_Z, q_Z_bivar, q_S_info = elbo_update(N, jnys, q_Z, lambda, rho, U);
    elbo_value = elbo(jnys, q_S_info, p_param_Z, q_Z, q_Z_bivar, U);
    println("[Iteration $iter]: ELBO = $elbo_value, rho = $rho")

  end

  return(q_Z, q_Z_bivar)

end
