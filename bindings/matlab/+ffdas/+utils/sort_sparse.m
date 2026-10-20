function [order, sparse_indices, offsets, weights] = sort_sparse(sparse_indices, offsets, weights)
% SORT_SPARSE Reorder a DAS_SPARSE problem so its cost follows the nonzero entries.
%   [ORDER, SPARSE_INDICES, OFFSETS, WEIGHTS] = SORT_SPARSE(SPARSE_INDICES,
%   OFFSETS, WEIGHTS) puts each target's nonzero entries first, in ascending
%   event order, and orders the targets by those event lists (keeping their
%   own order within a group, so neighbours stay neighbours).
%
%   DAS_SPARSE runs 32 targets together, at the pace of the one with the
%   most nonzero entries, and reads fastest when they read the same event
%   at the same time. Measured on 16 diverging transmits: 1.7-2.5x faster
%   than grid order, and faster than one call per transmit. Compute it once
%   per geometry.
%
%   SPARSE_INDICES, OFFSETS, WEIGHTS: (targets, n), as passed to DAS_SPARSE
%   (reshape the spatial dimensions to one first). ORDER: (targets, 1).
%   Pass DSTPOS(:, ORDER) with the reordered arrays, and put the result
%   back with IMAGE(ORDER, :) = RESULT.

    n = size(sparse_indices, 2);
    % an entry that weighs 0 is padding: it sorts after every event
    key = double(gather(sparse_indices));
    key(gather(weights) == 0) = inf;
    [key, entries] = sort(key, 2);          % stable
    [~, order] = sortrows(key);             % stable
    entries = entries(order, :);
    lin = sub2ind(size(key), repmat(order, 1, n), entries);
    sparse_indices = sparse_indices(lin);
    offsets = offsets(lin);
    weights = weights(lin);
end
