% Main script demonstrating comprehensive relationship patterns

% Call setup script first (execution order test)
run('setup_script.m');

% Use configuration from setup script (cross-scope access)
if exist('config', 'var')
    debug_enabled = config.debug_mode;
    max_iters = config.max_iterations;
end

% Define script-level variables
x = 10;
y = x + 5;
z = y * 2;

% Call functions
result1 = helper_function(x);
[modified_x, modified_y] = modify_variables(x, y);

% Call another script
run('helper_script.m');

% Use variables in calculations
final_result = modified_x + modified_y + z;

% Display results
display_results(final_result, result1);
