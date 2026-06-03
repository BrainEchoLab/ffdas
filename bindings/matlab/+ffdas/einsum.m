function y = einsum(subscripts, a, x)
% EINSUM Binary tensor contraction using Einstein summation notation.
%   Y = EINSUM(SUBSCRIPTS, A, X) contracts tensors A and X according to
%   the subscript string. Supports explicit output modes (e.g. 'ij,jk->ik')
%   and implicit mode where repeated indices are summed and unique indices
%   are kept in order of first appearance.
%
%   Both operands must have the same dtype. Scalar outputs (all indices
%   contracted) are squeezed to a scalar.
%
%   Examples:
%     C = ffdas.einsum('ij,jk->ik', A, B);  % matrix multiplication
%     C = ffdas.einsum('bij,bjk->bik', A, B);  % batched matmul
%     C = ffdas.einsum('i,j->ij', a, b);  % outer product
%
%   Inputs:
%     SUBSCRIPTS - Subscript string (char).
%     A          - First operand (gpuArray).
%     X          - Second operand (gpuArray).
%
%   Output:
%     Y - Contracted result (gpuArray).

    arguments
        subscripts char
        a gpuArray  
        x gpuArray
    end
    
    % parse subscripts to extract input and output modes
    sides = split(subscripts, '->');
    
    if length(sides) == 1
        % no '->' found, infer output modes
        modes = split(sides{1}, ',');
        if length(modes) ~= 2
            error('subscripts must contain exactly one '','' (got ''%s'')', subscripts);
        end
        am = modes{1};
        xm = modes{2};
        
        % infer output modes: include modes that appear exactly once
        all_modes = [am, xm];
        mode_count = containers.Map();
        for i = 1:length(all_modes)
            mode = all_modes(i);
            if isKey(mode_count, mode)
                mode_count(mode) = mode_count(mode) + 1;
            else
                mode_count(mode) = 1;
            end
        end
        
        % output modes are those that appear exactly once, in order of first appearance
        ym = '';
        seen = containers.Map();
        for i = 1:length(all_modes)
            mode = all_modes(i);
            if mode_count(mode) == 1 && ~isKey(seen, mode)
                ym = [ym, mode];
                seen(mode) = true;
            end
        end
    
    elseif length(sides) == 2
        % explicit output modes provided
        modes = split(sides{1}, ',');
        if length(modes) ~= 2
            error('subscripts must contain exactly one '','' (got ''%s'')', subscripts);
        end
        am = modes{1};
        xm = modes{2};
        ym = sides{2};
    else
        error('subscripts must contain at most one ''->'''' (got ''%s'')', subscripts);
    end
    
    % validate inputs have same datatype
    if ~strcmp(class(a), class(x))
        error('Input tensors must have the same datatype');
    end
    
    % convert mode characters to ASCII values
    am_int = int32(am) - int32('a');
    xm_int = int32(xm) - int32('a');
    ym_int = int32(ym) - int32('a');
    
    % handle scalar output case
    scalar_output = isempty(ym);
    if scalar_output
        % add a dummy mode and dimension for scalar output
        dummy_mode = max([am_int, xm_int]) + 1;
        ym_int = dummy_mode;
        
        % add dummy dimension of size 1 to x
        x = reshape(x, [size(x), 1]);
        xm_int = [xm_int, dummy_mode];
    end
    
    h = ffdas.core.get_handle();

    y = ffdas.core.ffdas_contraction(h, x, xm_int, a, am_int, ym_int);
    
    % remove dummy dimension for scalar output
    if scalar_output
        y = squeeze(y);
    end
end
