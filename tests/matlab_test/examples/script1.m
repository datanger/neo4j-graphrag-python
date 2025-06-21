% Script called by test_calls function
disp('This is script1.m executing');
% Script calling a function (Script -> Function)
result = helper_script_function(20);
disp(['Script function result: ', num2str(result)]);

% Script calling another script (Script -> Script)
script2;
