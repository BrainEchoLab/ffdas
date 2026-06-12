function count = get_cuda_device_count()
% GET_CUDA_DEVICE_COUNT Get the number of available CUDA devices.
%
%   Output:
%     COUNT - Number of CUDA devices (int).

    count = ffdas.core.ffdas_device_count();
end
