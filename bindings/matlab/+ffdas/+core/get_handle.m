function h = get_handle()
    persistent guard
    if isempty(guard)
        guard = ffdas.core.Handle();
    end
    h = guard.h;
end
