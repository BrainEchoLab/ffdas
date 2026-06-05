function h = get_handle()
    persistent handles
    if isempty(handles)
        handles = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    end
    device = int32(ffdas.core.ffdas_device_get());
    if ~isKey(handles, device)
        handles(device) = ffdas.core.Handle();
    end
    h = handles(device).h;
end
