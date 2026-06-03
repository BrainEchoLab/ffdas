classdef (Hidden) Handle < handle
    properties
        h uint64 = uint64(0)
    end
    methods
        function obj = Handle()
            obj.h = ffdas.core.ffdas_create();
        end
        function delete(obj)
            if obj.h ~= uint64(0)
                ffdas.core.ffdas_destroy(obj.h);
                obj.h = uint64(0);
            end
        end
    end
end
