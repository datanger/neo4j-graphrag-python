% Setup script for execution order testing
% This script defines configuration variables that will be used by other scripts

% Configuration variables
config = struct();
config.debug_mode = true;
config.max_iterations = 100;
config.tolerance = 1e-6;

% Global configuration that can be shared
global shared_config;
shared_config = config;

% Initialize some base variables
base_value = 42;
setup_complete = true;

fprintf('Setup script completed. Configuration initialized.\n'); 