function out = einsum(subscripts, a, b)
% EINSUM Binary tensor contraction using Einstein summation notation.
%   OUT = EINSUM(SUBSCRIPTS, A, B) contracts tensors A and B according to
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
%     B          - Second operand (gpuArray).
%
%   Output:
%     OUT - Contracted result (gpuArray).

    arguments
        subscripts char
        a gpuArray  
        b gpuArray
    end
    
    sides = split(subscripts, '->');
    
    if length(sides) == 1
        modes = split(sides{1}, ',');
        if length(modes) ~= 2
            error('subscripts must contain exactly one '','' (got ''%s'')', subscripts);
        end
        am = modes{1};
        bm = modes{2};
        
        all_modes = [am, bm];
        mode_count = containers.Map();
        for i = 1:length(all_modes)
            mode = all_modes(i);
            if isKey(mode_count, mode)
                mode_count(mode) = mode_count(mode) + 1;
            else
                mode_count(mode) = 1;
            end
        end
        
        outm = '';
        seen = containers.Map();
        for i = 1:length(all_modes)
            mode = all_modes(i);
            if mode_count(mode) == 1 && ~isKey(seen, mode)
                outm = [outm, mode];
                seen(mode) = true;
            end
        end
    
    elseif length(sides) == 2
        modes = split(sides{1}, ',');
        if length(modes) ~= 2
            error('subscripts must contain exactly one '','' (got ''%s'')', subscripts);
        end
        am = modes{1};
        bm = modes{2};
        outm = sides{2};
    else
        error('subscripts must contain at most one ''->'''' (got ''%s'')', subscripts);
    end
    
    if ~strcmp(class(a), class(b))
        error('Input tensors must have the same datatype');
    end
    
    am_int = int32(am) - int32('a');
    bm_int = int32(bm) - int32('a');
    outm_int = int32(outm) - int32('a');
    
    scalar_output = isempty(outm);
    if scalar_output
        dummy_mode = max([am_int, bm_int]) + 1;
        outm_int = dummy_mode;
        
        b = reshape(b, [size(b), 1]);
        bm_int = [bm_int, dummy_mode];
    end
    
    h = ffdas.core.get_handle();

    out = ffdas.core.ffdas_contraction(h, a, am_int, b, bm_int, outm_int);
    
    if scalar_output
        out = squeeze(out);
    end
end
