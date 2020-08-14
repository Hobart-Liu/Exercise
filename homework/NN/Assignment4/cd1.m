function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
% error('not yet implemented');

      % this line is just to covert non-binary input (e.g. image data is between (0,1))
      % to binary input 
      visible_data = sample_bernoulli(visible_data);


      
      % lines below are really coded for CD-1. 

      h_prob = visible_state_to_hidden_probabilities(rbm_w, visible_data);
      h= sample_bernoulli(h_prob);
      
      positive_gradient = configuration_goodness_gradient(visible_data, h);
      
      v_prob = hidden_state_to_visible_probabilities(rbm_w, h);
      v = sample_bernoulli(v_prob);
      
      h_prob = visible_state_to_hidden_probabilities(rbm_w, v);
%      h = sample_bernoulli(h_prob);
      
%      negative_gradient = configuration_goodness_gradient(v, h);
       negative_gradient = configuration_goodness_gradient(v, h_prob);

% reason of comment out h and negative_gradient(v,h)       
       
% two alternatives: 1) h_prob -> h -> negative_gradient   2) h_prob -> negative_gradient directly. 
% the first alternative does not change theexpected value of the gradient estimate that CD-1 produces; 
% it only increases its variance. More variance means that we have to use a smaller learning rate, 
% and that means that it'll learn more slowly; in other words, we don't want more variance, especially 
% if it doesn't give us anything pleasant to compensate for that slower learning. 
 
      
      ret = positive_gradient - negative_gradient;
   
end
