classdef Timer < handle
    properties (Hidden)
        start_event uint64 = uint64(0)
        stop_event uint64 = uint64(0)
        running logical = false
        elapsed single = []
        h uint64 = uint64(0)
    end
    methods
        function obj = Timer()
            obj.start_event = ffdas.core.ffdas_event_create();
            obj.stop_event = ffdas.core.ffdas_event_create();
            obj.h = ffdas.core.get_handle();
        end
        function start(obj)
            if obj.running
                error('Timer is already running');
            end
            obj.running = true;
            ffdas.core.ffdas_event_record(obj.h, obj.start_event);
        end
        function stop(obj)
            if ~obj.running
                return;
            end
            ffdas.core.ffdas_event_record(obj.h, obj.stop_event);
            ffdas.core.ffdas_event_synchronize(obj.stop_event);
            obj.elapsed = ffdas.core.ffdas_event_elapsed_time(obj.start_event, obj.stop_event);
            obj.running = false;
        end
        function ms = elapsed_ms(obj)
            if isempty(obj.elapsed)
                error('Timer has not been stopped yet');
            end
            ms = obj.elapsed;
        end
        function delete(obj)
            if obj.start_event ~= uint64(0)
                ffdas.core.ffdas_event_destroy(obj.start_event);
                obj.start_event = uint64(0);
            end
            if obj.stop_event ~= uint64(0)
                ffdas.core.ffdas_event_destroy(obj.stop_event);
                obj.stop_event = uint64(0);
            end
        end
    end
end
