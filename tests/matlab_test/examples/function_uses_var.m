function result = function_uses_var()
    % This function demonstrates USES relationship with a global variable
    % Creates USES relationship with shared_var
    global shared_var;
    
    % Check if shared_var exists, if not initialize it
    if ~exist('shared_var', 'var') || isempty(shared_var)
        shared_var = 10;  % This would create a DEFINES relationship
    end
    
    % This usage creates a USES relationship with shared_var
    result = shared_var * 2;
    
    % Using another variable from another function
    % This creates USES relationship with helper_function
    helper_result = helper_function();
    
    % This creates USES relationship with helper_result
    disp(['Helper result: ' num2str(helper_result)]);
end

function res = helper_function()
    % This helper function is called by function_uses_var
    res = 42;
end
