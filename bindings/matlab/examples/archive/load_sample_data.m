function [params, data] = load_sample_data(data_dir, chunks)
    if ~exist(data_dir, "dir")
        error("Data directory not found: %s", data_dir);
    end
    
    param_file = fullfile(data_dir, "rf-data-parameters.h5");
    data_file = fullfile(data_dir, "rf-data.bin");
    
    % load acquisition parameters
    nchannels = h5read(param_file, "/nchannels");
    nrepeats = h5read(param_file, "/nrepeats");
    ntransmissions = h5read(param_file, "/ntransmissions");
    nsamples = h5read(param_file, "/nsamples");
    
    chunk_size = [nsamples, ntransmissions, nrepeats, nchannels];
    
    channel_map = h5read(param_file, "/channel_map");
    channel_map = gpuArray(int32(channel_map(:)));  % convert to column vector and int32
    
    demod_frequency = h5read(param_file, "/demod_frequency");
    center_frequency = h5read(param_file, "/center_frequency");
    sound_speed = h5read(param_file, "/sound_speed");
    
    % divide by two because we interpret consecutive samples as (real, complex) IQ pairs
    sampling_frequency = h5read(param_file, "/sampling_frequency") / 2;
    
    % convert from mm to m and transpose
    focus = gpuArray(single(1e-3 * h5read(param_file, "/tx_focus")'));
    channel_pos = gpuArray(single(1e-3 * h5read(param_file, "/group_position")'));
    
    % read binary data header
    fid = fopen(data_file, "rb");
    if fid == -1
        error("Could not open data file: %s", data_file);
    end
    
    header = fread(fid, 4, "int64");
    header_bytes = header(2);
    max_chunks = header(3);
    
    chunks = min(chunks, max_chunks);
    
    % read the raw data
    fseek(fid, header_bytes, "bof");
    total_elements = chunks * prod(chunk_size);
    raw_data = fread(fid, total_elements, "*int16");
    fclose(fid);
    
    % reshape and convert to complex
    data = reshape(raw_data, [chunk_size, chunks]);
    data = complex(data(1:2:end,:,:,:,:), data(2:2:end,:,:,:,:));
    data = gpuArray(data);
    
    fprintf("data shape: [%s]\n", num2str(size(data)));
    fprintf("# channels: %d\n", length(channel_pos));
    
    params = struct();
    params.channel_map = channel_map;
    params.channel_pos = channel_pos;
    params.demod_frequency = demod_frequency;
    params.center_frequency = center_frequency;
    params.sampling_frequency = sampling_frequency;
    params.sound_speed = sound_speed;
    params.focus = focus;
end
