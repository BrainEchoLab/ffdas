%% FastDAS einsum example
% Demonstrates Einstein summation notation for tensor contractions

addpath('../bin');

% Create some sample GPU arrays
A = gpuArray.rand(4, 3, 'single');
B = gpuArray.rand(3, 5, 'single');
C = gpuArray.rand(4, 4, 'single');

fprintf('FastDAS einsum examples:\n');
fprintf('A: %dx%d, B: %dx%d, C: %dx%d\n\n', size(A), size(B), size(C));

% Example 1: Matrix multiplication
fprintf('1. Matrix multiplication: A * B\n');
fprintf('   einsum(''ij,jk->ik'', A, B)\n');
result1 = ffdas.einsum('ij,jk->ik', A, B);
matlab_result1 = A * B;
fprintf('   Result size: %dx%d\n', size(result1));
fprintf('   Matches MATLAB A*B: %s\n\n', string(norm(gather(result1 - matlab_result1)) < 1e-6));

% % Example 3: Element-wise multiplication
% fprintf('3. Element-wise multiplication: A .* A\n');
% fprintf('   einsum(''ij,ij->ij'', A, A)\n');
% result3 = ffdas.einsum('ij,ij->ij', A, A);
% matlab_result3 = A .* A;
% fprintf('   Result size: %dx%d\n', size(result3));
% fprintf('   Matches MATLAB A.*A: %s\n\n', string(norm(gather(result3 - matlab_result3)) < 1e-6));

% % Example 4: Sum over columns (reduce last dimension)
% fprintf('4. Sum over columns: sum(A, 2)\n');
% fprintf('   einsum(''ij->i'', A, [])\n');
% result4 = ffdas.einsum('ij->i', A, []);
% matlab_result4 = sum(A, 2);
% fprintf('   Result size: %dx1\n', length(result4));
% fprintf('   Matches MATLAB sum(A,2): %s\n\n', string(norm(gather(result4 - matlab_result4)) < 1e-6));

% Example 5: Outer product
x = gpuArray.rand(3, 1, 'single');
y = gpuArray.rand(4, 1, 'single');
fprintf('5. Outer product: x * y''\n');
fprintf('   einsum(''i,j->ij'', x, y)\n');
result5 = ffdas.einsum('ik,jk->ij', x, y);
matlab_result5 = x * y';
fprintf('   Result size: %dx%d\n', size(result5));
fprintf('   Matches MATLAB x*y'': %s\n', string(norm(gather(result5 - matlab_result5)) < 1e-6));

fprintf('\nAll einsum operations completed successfully!\n');
