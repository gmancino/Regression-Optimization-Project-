%-----------------------------------
% Relevant Project Computations
%-----------------------------------
% NOTE: The results generated from
% this script will produce the results
% on the report for e=10^-2
%-----------------------------------
%% Load the Data and Set Initial Parameters
df_g = load('gisette.mat');
df_s = load('spamData.mat');

% Separate into training and testing for gisette data
g_xtrain = df_g.Xtrain;
g_ytrain = df_g.ytrain;
g_xtest = df_g.Xtest;
g_ytest = df_g.ytest;

% Separate into training and testing for spam data
s_xtrain = df_s.Xtrain;
s_ytrain = df_s.ytrain;
s_xtest = df_s.Xtest;
s_ytest = df_s.ytest;

%%%%%%%%%%%%%%%
maxiter = 10000;
tol = 10^(-2);
%%%%%%%%%%%%%%%
lam1 = 0.001;
lam2 = 0.001;

%% True Optimal Objective
tol_opt = 10^(-16);

% Spam Data
[w_n_s_opt, b_n_s_opt, iter_n_s_opt, hist_obj_n_s_opt] = Newton(s_xtrain, s_ytrain,...
    zeros(57,1), 0, lam1, lam2, maxiter, tol_opt);

% Gisette Data
[w_n_g_opt, b_n_g_opt, iter_n_g_opt, hist_obj_n_g_opt] = Newton(g_xtrain, g_ytrain,...
    zeros(5000,1), 0, lam1, lam2, maxiter, tol_opt);

%% Test Methods on Spam Data

% Steepest Gradient Descent
t0 = tic;
[w_sd_s, b_sd_s, iter_sd_s, hist_obj_sd_s] = SteepGD(s_xtrain, s_ytrain,...
    zeros(57,1), 0, lam1, lam2, maxiter, tol);
time_sd_s = toc(t0);
% Newton's Method
t0 = tic;
[w_n_s, b_n_s, iter_n_s, hist_obj_n_s] = Newton(s_xtrain, s_ytrain,...
    zeros(57,1), 0, lam1, lam2, maxiter, tol);
time_n_s = toc(t0);
% Stochastic Gradient Descent
t0 = tic;
[w_sgd_s, b_sgd_s, iter_sgd_s, hist_obj_sgd_s] = SGD(s_xtrain, s_ytrain,...
    zeros(57,1), 0, lam1, lam2, maxiter/25, 100);
time_sgd_s = toc(t0);

%% Test Methods on Gisette Data

% Steepest Gradient Descent
t0 = tic;
[w_sd_g, b_sd_g, iter_sd_g, hist_obj_sd_g] = SteepGD(g_xtrain, g_ytrain,...
    zeros(5000,1), 0, lam1, lam2, maxiter, tol);
time_sd_g = toc(t0);
% Newton's Method
t0 = tic;
[w_n_g, b_n_g, iter_n_g, hist_obj_n_g] = Newton(g_xtrain, g_ytrain,...
    zeros(5000,1), 0, lam1, lam2, maxiter, tol);
time_n_g = toc(t0);
% Stochastic Gradient Descent
t0 = tic;
[w_sgd_g, b_sgd_g, iter_sgd_g, hist_obj_sgd_g] = SGD(g_xtrain, g_ytrain,...
    zeros(5000,1), 0, lam1, lam2, maxiter/10, 100);
time_sgd_g = toc(t0);

%% Classify New Points, Plot Objective, Print Accuracy of Spam Data

% Classify New Points
y_sd_s = ClassLR(s_xtest, w_sd_s, b_sd_s);
y_n_s = ClassLR(s_xtest, w_n_s, b_n_s);
y_sgd_s = ClassLR(s_xtest, w_sgd_s, b_sgd_s);
% Plots
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(hist_obj_sd_s-hist_obj_n_s_opt(length(hist_obj_n_s_opt)));
title('Objective History of Steepest GD on Spam Data');
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(hist_obj_n_s-hist_obj_n_s_opt(length(hist_obj_n_s_opt)));
title('Objective History of Newtons Method on Spam Data');
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(hist_obj_sgd_s-hist_obj_n_s_opt(length(hist_obj_n_s_opt)));
title('Objective History of Stochastic GD on Spam Data');
% Print Accuracy
fprintf('Accuracy of Steepest Gradient Descent on Spam Data: %2.2f%%.\n',...
    sum(y_sd_s==s_ytest)/length(s_ytest)*100)
fprintf('Accuracy of Newtons Method on Spam Data: %2.2f%%.\n',...
    sum(y_n_s==s_ytest)/length(s_ytest)*100)
fprintf('Accuracy of Stochastic Gradient Descent on Spam Data: %2.2f%%.\n',...
    sum(y_sgd_s==s_ytest)/length(s_ytest)*100)
fprintf('\n')

%% Classify New Points, Plot Objective, Print Accuracy of Gisette Data

% Classify New Points
y_sd_g = ClassLR(g_xtest, w_sd_g, b_sd_g);
y_n_g = ClassLR(g_xtest, w_n_g, b_n_g);
y_sgd_g = ClassLR(g_xtest, w_sgd_g, b_sgd_g);
% Plots
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(hist_obj_sd_g-hist_obj_n_g_opt(length(hist_obj_n_g_opt)));
title('Objective History of Steepest GD on Gisette Data');
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(hist_obj_n_g-hist_obj_n_g_opt(length(hist_obj_n_g_opt)));
title('Objective History of Newtons Method on Gisette Data');
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
semilogy(hist_obj_sgd_g-hist_obj_n_g_opt(length(hist_obj_n_g_opt)));
title('Objective History of Stochastic GD on Gisette Data');
% Print Accuracy
fprintf('Accuracy of Steepest Gradient Descent on Gisette Data: %2.2f%%.\n',...
    sum(y_sd_g==g_ytest)/length(g_ytest)*100)
fprintf('Accuracy of Newtons Method on Gisette Data: %2.2f%%.\n',...
    sum(y_n_g==g_ytest)/length(g_ytest)*100)
fprintf('Accuracy of Stochastic Gradient Descent on Gisette Data: %2.2f%%.\n',...
    sum(y_sgd_g==g_ytest)/length(g_ytest)*100)
fprintf('\n')
