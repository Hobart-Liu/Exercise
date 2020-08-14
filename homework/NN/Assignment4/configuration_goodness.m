function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
%    error('not yet implemented');

  G = mean(diag(hidden_state' * rbm_w * visible_state));
  
  % h' * w * v only good for h,v as vector. 
  % if we have a batch of h and v, h,v must be corresponding to each other.
  % in other word h[][1] must work together with v[][1]
  % Reason: in one configuration, h and v are tightly hooking together
  % there is no meaning to calculate h[][1] * v[][2]. 
  % Hereby we use diag to just pick diagnous of matrix h'*W*v to pick corresponding items. 

end
