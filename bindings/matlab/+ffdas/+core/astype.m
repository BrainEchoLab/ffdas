function arr = astype(arr, dtype)
%ASTYPE Cast arr to dtype if needed, with a warning.
%   Internal helper. Warns with ID 'ffdas:downcast'.
%
%   Control behaviour with MATLAB's warning system:
%     warning('off',   'ffdas:downcast')  % suppress warnings
%     warning('on',    'ffdas:downcast')  % restore warnings (default)
%     warning('error', 'ffdas:downcast')  % turn into an error

    arguments
        arr
        dtype
    end

    if isempty(arr)
        return;
    end

    if isa(arr, 'gpuArray')
        actual_class = classUnderlying(arr);
    else
        actual_class = class(arr);
    end

    if strcmp(actual_class, dtype)
        return;
    end

    warning('ffdas:downcast', ...
        'input of class ''%s'' will be downcast to ''%s''', ...
        actual_class, dtype ...
    );
    arr = cast(arr, dtype);
end
