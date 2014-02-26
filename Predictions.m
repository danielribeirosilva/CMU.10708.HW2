%HW2 Daniel Ribeiro Silva
%drsilva
%Predictions


%load data
load('hmm_params.mat');

%constants from problem
Q = 3;   %total hidden states
O = 5;   %total options

all_samples = zeros(O,size(100:127,2));
correctness = zeros(1,size(100:127,2));

for T = 100:127
    
    %get probability distribution prediction for Zt at t=T-1
    pred_Z_T_minusOne = forwardAlgorithm(Q,T,price_change,prior,emission,transition);
    
    %use transition matrix to get distribution for Z_T
    pred_Z_T = transition'*pred_Z_T_minusOne;
    pred_Z_T = pred_Z_T  / sum(pred_Z_T );
    
    %emission probability
    emission_prob = emission'*pred_Z_T;
    
    %sample from multinomial
    n_samples = 100;
    samples = mnrnd(n_samples,emission_prob)';
    
    %sample X_T
    all_samples(:,T-100+1) = samples;
    
    %correctness
    correctness(T-100+1) = samples(price_change(T+1))/n_samples;
    
end

plot(101:128,correctness);

disp(mean(correctness));
disp(var(correctness));
