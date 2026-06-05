function set_cuda_device(device)
% SET_CUDA_DEVICE Set the cuda device used on the current host thread.
%
%   Inputs:
%     DEVICE    - Device index (int).

    arguments 
        device int32
    end

    ffdas.core.ffdas_device_set(device);
end
