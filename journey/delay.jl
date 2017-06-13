##########################################################################################
# Basic delay model
#
# Functions in this file implement the basic delay model of Nicolo and Silva (2017),
# "Tomography of the London Underground: a Scalable Model for Origin-Destionation Data".
#
# NOTICE: THIS IS JUST THE BEGINNING OF A CONVERSION OF THE ORIGINAL MATLAB SCRIPTS!
# NOT FINALIZED!

"""
  relaxed_path_sel(n::Integer, U::Underground)


Generate for each pair of stations in the Underground variable ``U''
a set of up to ``n`` paths according to how long they are.  
"""
function relaxed_path_sel(n::Integer, U::Underground)

end

function single_line_path(i::Integer, j::Integer, U::Underground)

  best_length = Inf;

  for n = 1:U.num_lines
  
    A=graph(U.SingleNetworks{n});
    Path=shortestpath(A,i,j);
    if isempty(Path)==0
        L=length(Path);
        if L<Best
            SingleLinePath=Path;
            Best=L;
        end
    end
  end

  if isinf(best_length)
    return([])
  end
  return(path_choice)

end
