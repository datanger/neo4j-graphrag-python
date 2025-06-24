%% Test Entry Script for Cross-Scope Analysis
% This script calls other scripts to test cross-scope relationship generation

% Load BERT model
mdl = bert;

% Call other scripts
bert;
predictMaskedToken;
finbert;

% Use variables from called scripts
result = mdl;
tokenizer = mdl.Tokenizer;
