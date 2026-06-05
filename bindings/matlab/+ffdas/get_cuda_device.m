function device = get_cuda_device()
% GET_CUDA_DEVICE Get the cuda device used on the current host thread.
%
%   Output:
%     DEVICE    - Index of the current device (int).

    device = ffdas.core.ffdas_device_get();
end
