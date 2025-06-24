%% Test Script for Script -[USES]-> Variable Relationship
% This script demonstrates how scripts can use variables defined in other scripts
% This should generate Script -[USES]-> Variable relationships

% Load BERT model (this creates a variable 'mdl')
mdl = bert;

% Use the model variable in calculations
modelSize = size(mdl.Parameters.Weights.embedding.weight);
disp(['Model embedding size: ', num2str(modelSize)]);

% Use tokenizer from the model
tokenizer = mdl.Tokenizer;
vocabSize = tokenizer.VocabularySize;
disp(['Vocabulary size: ', num2str(vocabSize)]);

% Create some test data using the model
testText = "Hello world";
tokens = tokenize(tokenizer, testText);
encodedData = encodeTokens(tokenizer, tokens);

% Use the encoded data for prediction
predictions = predictMaskedToken(mdl, testText);

% Display results
fprintf('Input text: %s\n', testText);
fprintf('Number of tokens: %d\n', length(tokens{1}));
fprintf('Encoded data size: %s\n', mat2str(size(encodedData{1})));

% Use variables from other scripts (if they exist)
% This demonstrates cross-script variable usage
if exist('modelSize', 'var')
    fprintf('Model size from previous calculation: %s\n', mat2str(modelSize));
end

% Create a simple calculation using multiple variables
totalParameters = prod(modelSize) + vocabSize;
fprintf('Total parameters estimate: %d\n', totalParameters);
