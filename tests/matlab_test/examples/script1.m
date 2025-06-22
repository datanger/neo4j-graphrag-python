% Script called by test_calls function
disp('This is script1.m executing');
% Script calling a function (Script -> Function)
result = helper_script_function(20);
disp(['Script function result: ', num2str(result)]);

% Script calling another script (Script -> Script)
script2;

% Script1 calls Script2 and passes variables

val1 = 5;
run('script2.m');
val2 = val1 + val_from_script2;  % This uses a variable defined in script2.m

% Additional cross-scope usage
final_result = val_from_script2 * 2;  % Another cross-scope usage
