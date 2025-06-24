% Complex scope test script
% Tests multiple scripts with overlapping variable names and complex dependencies

% Script-level variables with common names
x = 1;
y = 2;
z = 3;
data = [x, y, z];

% Call another script that also defines x, y, z
run('helper_script.m');

% Use variables from both scopes
combined_data = [x, y, z, data];

% Function with same variable names
function process_data(x, y, z)
    % Function parameters (different scope)
    local_x = x * 2;
    local_y = y * 3;
    local_z = z * 4;
    
    % Local variables with same names as script
    x = local_x;
    y = local_y;
    z = local_z;
    
    % Return processed data
    result = [x, y, z];
end

% Call function with parameters
processed = process_data(10, 20, 30);

% Use variables after function call
final_x = x;
final_y = y;
final_z = z;

% Nested scope test
function outer_function()
    outer_var = 100;
    
    function inner_function()
        inner_var = 200;
        % Access outer function variable
        combined = outer_var + inner_var;
    end
    
    inner_function();
end

% Call nested function
outer_function(); 