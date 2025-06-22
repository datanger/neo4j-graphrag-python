% This script demonstrates USES relationships with variables
% It will create USES relationships with all the variables it uses

% Load configuration data from a .mat file
% This creates USES relationships with any variables loaded from the file
if exist('config_data.mat', 'file')
    loaded_data = load('config_data.mat');
    config = loaded_data.config;  % USES relationship with config
else
    % Create default config if file doesn't exist
    config = struct();
    config.parameters = struct('threshold', 0.5, 'max_iter', 100);
    save('config_data.mat', 'config');
end

% Call a function that returns multiple outputs
% This creates USES relationships with all output variables
[data, metadata] = load_data('sample.dat');  % USES with data and metadata

% Process the data using the config
% Creates USES relationship with result and all variables used in process_data
result = process_data(data, config.parameters);

% Display results with formatting
fprintf('Processing complete. Result: %.2f\n', result);

% Nested function that also creates USES relationships
function output = process_data(input_data, params)
    % This function creates USES relationships with input_data and params
    output = input_data * params.threshold;

    % Using a variable from the parent workspace
    % This creates a USES relationship with config
    if isfield(config, 'debug_mode') && config.debug_mode
        disp('Debug mode: Processing data...');
    end
end
