function y = take_along_axis(x, indices, axis)
% TAKE_ALONG_AXIS   Pick elements from X along axis AXIS.
%    Y = TAKE_ALONG_AXIS(X, INDICES, AXIS) returns an array Y the same size as
%    INDICES where
%       Y(i1,...,i_{axis-1}, j, i_{axis+1},...,iN) = ...
%           X(i1,...,i_{axis-1}, INDICES(i1,...,i_{axis-1}, j, i_{axis+1},...,iN), i_{axis+1},...,iN)
%
%    IDX must be integers in 1..size(X,dim).

    sz_x = size(x);
    ndim = max(ndims(x), axis);
    sz_x(end+1:ndim) = 1;  % pad with ones if dim > ndims(x)
    sz_y = size(indices);

    expected_y = sz_x;
    expected_y(axis) = sz_y(axis);   % along axis, y is as big as indices

    if ~isequal(sz_y, expected_y)
        error('take_along_axis:SizeMismatch', ...
              'INDICES must have size [%s] where [%s] except along AXIS', ...
              strjoin(string(expected_y),','), ...
              strjoin(string(sz_x),','));
    end

    % col-maj strides
    stride = [1, cumprod(sz_x(1:end-1))];

    % start with the contribution from the axis-th index
    lin = (indices - 1) * stride(axis);

    % add contributions of all the other dimensions
    for k = 1:ndim
        if k==axis
            continue; 
        end

        % build a “subscript array” for dim k:
        %   subs{k}(i1,...,iN) = ik
        subsz = ones(1, ndim);
        subsz(k) = sz_x(k);
        subs = reshape(1:sz_x(k), subsz);

        % now replicate that to match y’s size
        reps = sz_y ./ subsz;
        subs = repmat(subs, reps);

        % add its linear‐index contribution
        lin = lin + (subs - 1) * stride(k);
    end

    y = x(lin + 1);
end
