%% Entry Script for Testing Script -[USES]-> Variable Relationships
% This script calls other scripts and uses variables from them

% First, run the simple script to create some variables
simple_script;

% Now use variables that were created in simple_script
% Note: In MATLAB, variables from scripts persist in the workspace
disp(['Using x from simple_script: ', num2str(x)]);
disp(['Using y from simple_script: ', num2str(y)]);
disp(['Using result from simple_script: ', num2str(result)]);

% Create new variables using the ones from simple_script
newResult = x * 2 + y * 3;
disp(['New calculation: ', num2str(newResult)]);

% Use variables in a more complex calculation
finalResult = result + newResult + z;
disp(['Final result: ', num2str(finalResult)]);

% Test with BERT model (if available)
try
    mdl = bert;
    disp('BERT model loaded successfully');

    % Use model variables
    if isfield(mdl, 'Tokenizer')
        tokenizer = mdl.Tokenizer;
        disp(['Tokenizer type: ', class(tokenizer)]);
    end

    if isfield(mdl, 'Parameters')
        params = mdl.Parameters;
        disp('Model parameters available');
    end
catch ME
    disp(['BERT model not available: ', ME.message]);
end

% Display workspace variables
disp('Workspace variables:');
whos
