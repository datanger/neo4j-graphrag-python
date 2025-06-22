% Script called by script1.m
disp('This is script2.m executing');
% Script calling a function (Script -> Function)
result = helper_script_function(30);
disp(['Script2 function result: ', num2str(result)]);

% Script2 defines a variable to be used by Script1

val_from_script2 = 20;
