% Test script for global variable usage
% Demonstrates global keyword and cross-scope variable access

% Declare global variables
global global_counter;
global global_config;
global shared_data;

% Initialize global variables
global_counter = 0;
global_config = struct('enabled', true, 'timeout', 30);
shared_data = [];

% Local variables
local_var = 10;
temp_data = [1, 2, 3, 4, 5];

% Function that uses global variables
function update_global()
    global global_counter;
    global shared_data;
    
    % Modify global variables
    global_counter = global_counter + 1;
    shared_data = [shared_data, global_counter];
    
    % Local variable in function
    local_func_var = 20;
end

% Function that declares and uses global variables
function process_globals()
    global global_config;
    global new_global_var;
    
    % Declare a new global variable
    new_global_var = 'test_value';
    
    % Use existing global variable
    if global_config.enabled
        result = 'enabled';
    else
        result = 'disabled';
    end
end

% Call functions that use global variables
update_global();
process_globals();

% Use global variables in main script
final_counter = global_counter;
final_config = global_config; 