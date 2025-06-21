% External script called by test cases

% This script is called by both another script and a function
fprintf('External script executed\n');

% Define some variables in the script scope
script_var1 = 42;
script_var2 = 'Hello from external script';

% Call an external function
result = multiply_by_two(7);

% Use the result to avoid unused variable warning
fprintf('Result: %d\n', result);
