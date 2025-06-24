%% Simple Script for Testing Variable Usage
% This is a script (not a function) that uses variables

% Define some variables
x = 10;
y = 20;
z = x + y;

% Use variables in calculations
result = x * y + z;
disp(['Result: ', num2str(result)]);

% Use variables in conditional statements
if x > 5
    message = 'x is greater than 5';
    disp(message);
end

% Use variables in loops
for i = 1:x
    disp(['Iteration ', num2str(i), ' of ', num2str(x)]);
end

% Use variables in array operations
array = [x, y, z, result];
disp(['Array: ', mat2str(array)]);

% Use variables in function calls
maxValue = max(array);
minValue = min(array);
disp(['Max: ', num2str(maxValue), ', Min: ', num2str(minValue)]);
