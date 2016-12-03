  % model.input_to_hid  <number of hidden units> x <number of inputs i.e. 256>   It contains the weights from the input units to the hidden units.
  % model.hid_to_class  <number of classes i.e. 10> x <number of hidden units>   It contains the weights from the hidden units to the softmax units.
  % data.inputs         <number of inputs i.e. 256> x <number of data cases>     Each column describes a different data case. 
  % data.targets        <number of classes i.e. 10> x <number of data cases>     Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.


  
  %from_data_file =
  %
  %scalar structure containing the fields:
  %
  %  data =
  %
  %    scalar structure containing the fields:
  %
  %      training =
  %
  %        scalar structure containing the fields:
  %
  %          inputs: 256x1000 matrix
  %          targets: 10x1000 matrix
  %
  %      validation =
  %
  %        scalar structure containing the fields:
  %
  %          targets: 10x1000 matrix
  %          inputs: 256x1000 matrix
  %
  %      test =
  %
  %        scalar structure containing the fields:
  %
  %          inputs: 256x9000 matrix
  %          targets: 10x9000 matrix
  
  

function a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size)
  warning('error', 'Octave:broadcast');
  if exist('page_output_immediately'), page_output_immediately(1); end
  more off;
  
  model = initial_model(n_hid);  %传入隐层的节点数，生成2个权重的矩阵,input_to_hid, hid_to_class, 定义见文件头
  
  from_data_file = load('data.mat'); %数据结构见文件头
  datas = from_data_file.data;
  
  n_training_cases = size(datas.training.inputs, 2);  % 获取训练数据总数
  if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end  % ???? 习题

  % optimization
  theta = model_to_theta(model);  %将矩阵还原成为向量
  momentum_speed = theta * 0;
  training_data_losses = [];
  validation_data_losses = [];
  if do_early_stopping,
    best_so_far.theta = -1; % this will be overwritten soon
    best_so_far.validation_loss = inf;
    best_so_far.after_n_iters = -1;
  end
  for optimization_iteration_i = 1:n_iters,
    model = theta_to_model(theta);
    
    training_batch_start = mod((optimization_iteration_i-1) * mini_batch_size, n_training_cases)+1;
    training_batch.inputs = datas.training.inputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    training_batch.targets = datas.training.targets(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient));
    momentum_speed = momentum_speed * momentum_multiplier - gradient;
    theta = theta + momentum_speed * learning_rate;

    model = theta_to_model(theta);
    training_data_losses = [training_data_losses, loss(model, datas.training, wd_coefficient)];
    validation_data_losses = [validation_data_losses, loss(model, datas.validation, wd_coefficient)];
    if do_early_stopping && validation_data_losses(end) < best_so_far.validation_loss,
      best_so_far.theta = theta; % this will be overwritten soon
      best_so_far.validation_loss = validation_data_losses(end);
      best_so_far.after_n_iters = optimization_iteration_i;
    end
    if mod(optimization_iteration_i, round(n_iters/10)) == 0,
      fprintf('After %d optimization iterations, training data loss is %f, and validation data loss is %f\n', optimization_iteration_i, training_data_losses(end), validation_data_losses(end));
    end
  end
  if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end % check again, this time with more typical parameters
  if do_early_stopping,
    fprintf('Early stopping: validation loss was lowest after %d iterations. We chose the model that we had then.\n', best_so_far.after_n_iters);
    theta = best_so_far.theta;
  end
  % the optimization is finished. Now do some reporting.
  model = theta_to_model(theta);
  if n_iters ~= 0,
    clf;
    hold on;
    plot(training_data_losses, 'b');
    plot(validation_data_losses, 'r');
    legend('training', 'validation');
    ylabel('loss');
    xlabel('iteration number');
    hold off;
  end
  datas2 = {datas.training, datas.validation, datas.test};
  data_names = {'training', 'validation', 'test'};
  for data_i = 1:3,
    data = datas2{data_i};
    data_name = data_names{data_i};
    fprintf('\nThe loss on the %s data is %f\n', data_name, loss(model, data, wd_coefficient));
    if wd_coefficient~=0,
      fprintf('The classification loss (i.e. without weight decay) on the %s data is %f\n', data_name, loss(model, data, 0));
    end
    fprintf('The classification error rate on the %s data is %f\n', data_name, classification_performance(model, data));
  end
end

function test_gradient(model, data, wd_coefficient)
  base_theta = model_to_theta(model);
  h = 1e-2;
  correctness_threshold = 1e-5;
  analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient));
  % Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.
  for i = 1:100,
    test_index = mod(i * 1299721, size(base_theta,1)) + 1; % 1299721 is prime and thus ensures a somewhat random-like selection of indices
    analytic_here = analytic_gradient(test_index);
    theta_step = base_theta * 0;
    theta_step(test_index) = h;
    contribution_distances = [-4:-1, 1:4];
    contribution_weights = [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280];
    temp = 0;
    for contribution_index = 1:8,
      temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances(contribution_index)), data, wd_coefficient) * contribution_weights(contribution_index);
    end
    fd_here = temp / h;
    diff = abs(analytic_here - fd_here);
    % fprintf('%d %e %e %e %e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here);
    if diff < correctness_threshold, continue; end
    if diff / (abs(analytic_here) + abs(fd_here)) < correctness_threshold, continue; end
    error(sprintf('Theta element #%d, with value %e, has finite difference gradient %e but analytic gradient %e. That looks like an error.\n', test_index, base_theta(test_index), fd_here, analytic_here));
  end
  fprintf('Gradient test passed. That means that the gradient that your code computed is within 0.001%% of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).\n');
end


% sigmoid 函数
function ret = logistic(input)
  ret = 1 ./ (1 + exp(-input));
end

function ret = log_sum_exp_over_rows(a)
  % This computes log(sum(exp(a), 1)) in a numerically stable way
  maxs_small = max(a, [], 1);
  maxs_big = repmat(maxs_small, [size(a, 1), 1]);
  ret = log(sum(exp(a - maxs_big), 1)) + maxs_small;
end

function ret = loss(model, data, wd_coefficient)

	 
  % Before we can calculate the loss, we need to calculate a variety of intermediate values, like the state of the hidden units.
  hid_input = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  hid_output = logistic(hid_input); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  class_input = model.hid_to_class * hid_output; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
  % The following three lines of code implement the softmax.
  % However, it's written differently from what the lectures say.
  % In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
  % What we do here is exactly equivalent (you can check the math or just check it in practice), but this is more numerically stable. 
  % "Numerically stable" means that this way, there will never be really big numbers involved.
  % The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Octave.
  % Octave isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.
  class_normalizer = log_sum_exp_over_rows(class_input); % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
  log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
  class_prob = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>
  
  
  % Cost Function = cross entropy + weight decay = classification_loss + wd_loss 
  classification_loss = -mean(sum(log_class_prob .* data.targets, 1)); % select the right log class probability using that sum; then take the mean over all data cases.
  wd_loss = sum(model_to_theta(model).^2)/2*wd_coefficient; % weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
  ret = classification_loss + wd_loss;
end

function ret = d_loss_by_d_model(model, data, wd_coefficient)

  % The returned object is supposed to be exactly like parameter <model>, 
  % i.e. it has fields ret.input_to_hid and ret.hid_to_class. 
  % However, the contents of those matrices are gradients 
  % (d loss by d model parameter), instead of model parameters.
  
  % FF
  
  % (each record of X1 has 256 units)
  % (assumption, say we have 20 unit in hidden layer). 
  % (we have 10 units in classification layer)
  
  % X1 = data.inputs  256 x 1000
  % W1 = model.input_to_hid  20 x 256
  % Z1 = W1 * X1 = hid_input  20 x 1000
  hid_input = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  % X2 = sigmoid(Z1) = hid_output   20 x 1000
  hid_output = logistic(hid_input); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  % W2 = model.hid_to_class 10 x 20
  % Z2 = W2 * X2 = class_input 10 x 1000
  class_input = model.hid_to_class * hid_output; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  % Y = softmax(Z2) = class_prob 10 x 1000
  
  % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. 
  % size: <1> by <number of data cases>
  class_normalizer = log_sum_exp_over_rows(class_input); 
  % log of probability of each class. 
  % size: <number of classes, i.e. 10> by <number of data cases>
  log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); 
  % probability of each class. Each column (i.e. each case) sums to 1. 
  % size: <number of classes, i.e. 10> by <number of data cases>
  class_prob = exp(log_class_prob); 
  
  
  % BP
  % Cost Function = cross entropy + weight decay = classification_loss + wd_loss 
 
  % d(CE)/d(W2) = d(CE)/d(Z2) x d(Z2)/d(X2) = (y-t) * X2'
  d_CE_d_W2 = (class_prob - data.targets) * hid_output';
  % d(WD)/d(W2) = wd_coeffecient*W2
  d_WD_d_W2 = wd_coefficient * model.hid_to_class;
  
  
  % d(CE)/d(W1) = d(CE)/d(Z2) x d(Z2)/d(X2) x d(X2)/d(Z1) x d(Z1)/d(W1) = (y-t) * W2 * X2 * (1-X2) * X1
  d_CE_d_W1 = model.hid_to_class' * (class_prob - data.targets) .* (hid_output - hid_output.^2) * data.inputs';
  % d_CE_d_W1 = ((class_prob - data.targets)' * model.hid_to_class * hid_output * ( ones(size(hid_output))- hid_output)')' * data.inputs';
  % d(WD)/d(W1) = wd_coeffecient*W1
  d_WD_d_W1 = wd_coefficient * model.input_to_hid;
  
  m = size(data.inputs,2);

  ret.input_to_hid = d_CE_d_W1/m + d_WD_d_W1;
  ret.hid_to_class = d_CE_d_W2/m + d_WD_d_W2;
  %ret.input_to_hid = model.input_to_hid * 0;
  %ret.hid_to_class = model.hid_to_class * 0;  
  
end


% 矩阵转换成向量
function ret = model_to_theta(model)
  % This function takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
  input_to_hid_transpose = transpose(model.input_to_hid);
  hid_to_class_transpose = transpose(model.hid_to_class);
  ret = [input_to_hid_transpose(:); hid_to_class_transpose(:)];
end


% 向量转换成矩阵
function ret = theta_to_model(theta)
  % This function takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
  n_hid = size(theta, 1) / (256+10);
  ret.input_to_hid = transpose(reshape(theta(1: 256*n_hid), 256, n_hid));
  ret.hid_to_class = reshape(theta(256 * n_hid + 1 : size(theta,1)), n_hid, 10).';
end


% 计算2个权重矩阵的参数总个数，为每个权重赋一个cos()的数字，模拟随机，方便作业的最后结果评判
% 调用teata_to_model 将向量转换成2个矩阵
function ret = initial_model(n_hid)
  n_params = (256+10) * n_hid;
  as_row_vector = cos(0:(n_params-1));
  ret = theta_to_model(as_row_vector(:) * 0.1); % We don't use random initialization, for this assignment. This way, everybody will get the same results.
end


% 找到分类错误所占的百分比
function ret = classification_performance(model, data)
  % This returns the fraction of data cases that is incorrectly classified by the model.
  hid_input = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  hid_output = logistic(hid_input); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  class_input = model.hid_to_class * hid_output; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
  
  % 由于softmax和最大值正相关，所以忽略了softmax的具体计算过程，只需要比较最大值出现的位置即可
  % dump 是最大值， choices, targets 则是最大值出现的位置。比较choices 和 targets 不一致即可
  [dump, choices] = max(class_input); % choices is integer: the chosen class, plus 1.
  [dump, targets] = max(data.targets); % targets is integer: the target class, plus 1.
  % choice~=targets 的比较返回一个向量 [0,1,0,1,1,1...]
  % 表示相同（0）或不同（1）
  % 返回值是取不同的总数总个数,即错误占的百分比
  ret = mean(double(choices ~= targets));
end
