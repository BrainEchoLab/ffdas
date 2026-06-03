addpath("../bin");

a = gpuArray.rand(128*64*64, 128, "single");

rank = 1;

fprintf("matlab eigfilter...");
tic();
[u, s, v] = svd(a, "econ");
b_matlab = u(:, rank:end) * s(rank:end, rank:end) * v(:, rank:end)';
delta = toc();
fprintf("done. (%.4f s)\n", delta);

% to avoid measuring the overhead time from matlab when loading a mex function
% for the first time, we run the computation once before
tmp = ffdas.eigfilter(a, rank);

fprintf("ffdas eigfilter...");
tic();
b_ffdas = ffdas.eigfilter(a, rank);
delta = toc();
fprintf("done. (%.4f s)\n", delta);

diff = abs(b_matlab - b_ffdas);

err_max = max(diff, [], "all");
err_mean = mean(diff, "all");
err_mse = mean(diff .^ 2, "all");
err_rmse = sqrt(err_mse);

fprintf("error: max=%.8f, mean=%.8f, mse=%.8f, rmse=%.8f\n", err_max, err_mean, err_mse, err_rmse);
