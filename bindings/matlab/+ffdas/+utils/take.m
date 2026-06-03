function y = take(x, k, axis)
% TAKE   Take the first k entries of x along dimension axis.
    n = ndims(x);
    subs = repmat({':'}, 1, n);  % build a list of “:” for every dimension
    subs{axis} = 1:k;  % replace the dim-th entry with 1:k
    % use comma-expansion to index
    y = x(subs{:});
end
