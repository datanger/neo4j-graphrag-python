% Test script with nested function definitions
% Demonstrates scope handling for nested functions

% Main script variables
main_var = 100;
outer_var = 200;

% Nested function definition
function nested_function()
    % Variables in nested function scope
    nested_var = 300;
    inner_var = 400;
    
    % Access to outer scope variables
    result = main_var + nested_var;
    
    % Nested function within nested function
    function deeply_nested()
        deep_var = 500;
        % Access to multiple scopes
        final_result = main_var + nested_var + deep_var;
    end
    
    % Call the deeply nested function
    deeply_nested();
end

% Call the nested function
nested_function();

% Use variables after function call
final_output = main_var + outer_var; 