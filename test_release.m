function test_release
%TEST_RELEASE  Extract MATLAB archives from dist/ and run examples.
%   matlab -batch "test_release"

    root = fileparts(mfilename("fullpath"));
    dist_dir = fullfile(root, "dist");

    zips = dir(fullfile(dist_dir, "ffdas-*-matlab-*.zip"));
    if isempty(zips)
        error("No MATLAB archives found in %s", dist_dir);
    end

    parallel.gpu.enableCUDAForwardCompatibility(true);

    passed = 0;
    failed = 0;

    for i = 1:numel(zips)
        zip_path = fullfile(dist_dir, zips(i).name);
        fprintf("\n============================================================\n");
        fprintf("%s", zips(i).name);
        fprintf("\n============================================================\n");

        tmp = fullfile(tempdir, "ffdas_matlab_test_" + string(i));
        if isfolder(tmp), rmdir(tmp, "s"); end
        mkdir(tmp);

        try
            unzip(zip_path, tmp);
            binding_dir = fullfile(tmp, "ffdas-matlab");
            example_dir = fullfile(binding_dir, "examples");
            addpath(binding_dir);

            % smoke test: call a function from the package
            fprintf("\n  smoke test: ");
            try
                dev = ffdas.get_cuda_device();
                fprintf("PASS (device %d)\n", dev);
                passed = passed + 1;
            catch ex
                fprintf("FAIL — %s\n", ex.message);
                failed = failed + 1;
            end

            examples = dir(fullfile(example_dir, "*.m"));
            for j = 1:numel(examples)
                name = examples(j).name(1:end-2);
                fprintf("\n  %s: ", name);

                workdir = fullfile(tmp, "work_" + string(name));
                mkdir(workdir);
                old_dir = cd(workdir);

                try
                    run_isolated(fullfile(example_dir, examples(j).name));
                    fprintf("PASS\n");
                    passed = passed + 1;
                catch ex
                    fprintf("FAIL — %s\n", ex.message);
                    failed = failed + 1;
                end

                cd(old_dir);
            end

            rmpath(binding_dir);
        catch ex
            fprintf("  ERROR: %s\n", ex.message);
            failed = failed + 1;
        end

        try rmdir(tmp, "s"); catch, end
    end

    fprintf("\n============================================================\n");
    fprintf("%d passed, %d failed", passed, failed);
    fprintf("\n============================================================\n");

    if failed > 0
        error("test_matlab_release:failed", "%d test(s) failed", failed);
    end
end


function run_isolated(script_path)
    run(script_path);
    close all;
    clear ffdas.core.get_handle;
end
