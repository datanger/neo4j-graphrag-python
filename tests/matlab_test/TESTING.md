# MATLAB Code Extractor Test Suite

This directory contains the test suite for the MATLAB code extractor component. The tests are designed to verify the functionality of the MATLAB code parser, relationship extraction, and schema compliance based on the comprehensive requirements defined in `neo4j_graphrag/experimental/components/code_extractor/matlab/requirements.py`.

## Test Structure

```
tests/matlab_test/
├── test_data/                  # Test case files
│   ├── main_script.m          # Main test script with comprehensive patterns
│   ├── helper_function.m       # Helper function with variables
│   ├── modify_variables.m      # Function that modifies variables
│   ├── helper_script.m         # Helper script
│   ├── display_results.m       # Display function
│   ├── setup_script.m          # Setup script for execution order testing
│   └── advanced/               # Advanced test cases
│       ├── nested_functions.m  # Nested function definitions
│       ├── global_variables.m  # Global variable usage
│       └── complex_scope.m     # Complex scope scenarios
├── test_matlab_extractor.py    # Unit tests and integration tests
├── test_full_pipeline.py       # Full pipeline integration tests
└── TESTING.md                  # This file
```

## Test Categories

### 1. Unit Tests (`test_matlab_extractor.py`)
- **Scope-specific variable testing**
- **Relationship pattern validation**
- **Schema compliance verification**
- **Execution order analysis**
- **Cross-scope variable access**
- **Advanced scope features**
- **Relationship properties validation**

### 2. Integration Tests (`test_full_pipeline.py`)
- **End-to-end pipeline testing**
- **Neo4j database integration**
- **Cross-file relationship processing**
- **Data persistence and retrieval**
- **Error handling and recovery**
- **Performance and scalability**
- **Real-world scenario simulation**

## Schema Requirements

### Node Types

1. **Function Node**
   - Properties: `name`, `file_path`, `line_range`, `description`, `parameters`, `returns`
   - Represents MATLAB function definitions

2. **Variable Node**
   - Properties: `name`, `file_path`, `scope_id`, `scope_type`, `line_range`
   - **Critical**: Variables are scope-specific with ID format `var_{name}_{scope_type}_{scope_id}`
   - Each script/function creates separate variable nodes even for same names

3. **Script Node**
   - Properties: `name`, `file_path`, `description`
   - Represents MATLAB script files

### Relationship Types

1. **CALLS** - Function/script calls another function/script
2. **USES** - Function/script uses a variable defined in another scope
3. **DEFINES** - Script defines a variable/function, function defines a variable
4. **ASSIGNED_TO** - Variable is assigned to another variable
5. **MODIFIES** - Function/script modifies a variable from another scope

### Required Patterns

The schema must support these relationship patterns:
- `(Function, CALLS, Function)`
- `(Function, CALLS, Script)`
- `(Script, CALLS, Function)`
- `(Script, CALLS, Script)`
- `(Function, USES, Variable)`
- `(Script, USES, Variable)`
- `(Function, DEFINES, Variable)`
- `(Script, DEFINES, Variable)`
- `(Script, DEFINES, Function)`
- `(Function, MODIFIES, Variable)`
- `(Script, MODIFIES, Variable)`
- `(Variable, ASSIGNED_TO, Variable)`

## Test Cases

### 1. Main Script (`main_script.m`)
- Defines script-level variables (x, y, z) with proper scope isolation
- Calls helper functions with parameter passing
- Demonstrates variable assignment chains (x → y → z)
- Includes script-to-script calls using `run()` function
- Shows variable usage in calculations
- Tests cross-scope variable access patterns

### 2. Helper Function (`helper_function.m`)
- Takes input parameters (not cross-scope access)
- Defines local variables with same names as main script (scope isolation)
- Returns a computed value
- Demonstrates function scope and parameter handling

### 3. Variable Modifier (`modify_variables.m`)
- Takes multiple input parameters
- Returns multiple modified values
- Contains internal variables with same names as inputs (scope isolation)
- Shows parameter passing and return value handling

### 4. Helper Script (`helper_script.m`)
- Defines its own variables (a, b, c, d)
- Calls helper functions
- Demonstrates script-level variable scope
- Shows script-to-function calls

### 5. Display Function (`display_results.m`)
- Takes variable number of arguments using `varargin`
- Demonstrates function calling with multiple arguments
- Shows basic output functionality

### 6. Setup Script (`setup_script.m`)
- Defines configuration variables
- Tests execution order analysis
- Demonstrates cross-scope variable access based on execution order
- Shows `USES` relationships with execution order properties

## Advanced Test Cases

### 7. Nested Functions (`advanced/nested_functions.m`)
- Tests nested function definitions within scripts
- Verifies scope handling for nested functions
- Tests variable access patterns in nested scopes

### 8. Global Variables (`advanced/global_variables.m`)
- Tests `global` keyword usage
- Verifies global variable relationship patterns
- Tests cross-scope global variable access

### 9. Complex Scope (`advanced/complex_scope.m`)
- Tests complex scope scenarios
- Multiple scripts with overlapping variable names
- Complex execution order dependencies

## Unit Tests (`test_matlab_extractor.py`)

### 1. Scope-Specific Variables Test
- Verifies that variables have correct scope information
- Checks that variable IDs follow the format `var_{name}_{scope_type}_{scope_id}`
- Ensures variables with the same name in different scopes have different IDs
- Validates scope types (script vs function)
- Tests scope isolation between scripts and functions

### 2. Relationship Patterns Test
- Verifies all required relationship types are present:
  - CALLS: Between scripts and functions
  - USES: For variable usage (including cross-scope)
  - DEFINES: For variable and function definitions
  - MODIFIES: For variable modifications
  - ASSIGNED_TO: For variable assignments
- Checks for required relationship patterns between different node types
- Validates relationship properties (line_number, usage_type, etc.)

### 3. Schema Compliance Test
- Validates that the schema is properly defined
- Checks for required node types (Function, Variable, Script)
- Verifies all required relationship types are defined
- Ensures minimum number of patterns are present
- Tests property type definitions

### 4. Execution Order Analysis Test
- Tests automatic detection of script dependencies
- Verifies topological sorting for execution order
- Tests cross-scope variable access based on execution order
- Validates `USES` relationships with execution order properties

### 5. Cross-Scope Variable Access Test
- Tests variable access between different scopes
- Verifies proper handling of parameter passing vs cross-scope access
- Tests assignment relationships between variables in different scopes
- Validates scope isolation and relationship creation

### 6. Advanced Scope Features Test
- Tests nested functions and their scope handling
- Verifies global variable relationship patterns
- Tests complex scope scenarios with multiple overlapping variables
- Validates advanced scope features implementation

### 7. Relationship Properties Test
- Validates that all relationships have required properties
- Tests property types and values
- Verifies relationship metadata completeness
- Tests property validation and conversion

## Integration Tests (`test_full_pipeline.py`)

### 1. Full Pipeline Integration Test
- Tests complete end-to-end pipeline execution
- Verifies data flow from extraction to Neo4j storage
- Tests cross-file relationship processing
- Validates data integrity throughout the pipeline

### 2. Neo4j Database Integration Test
- Tests connection to Neo4j database
- Verifies data writing and retrieval
- Tests database schema compliance
- Validates data persistence and consistency

### 3. Cross-File Relationship Test
- Tests processing of multiple files simultaneously
- Verifies cross-file relationship creation
- Tests execution order analysis across files
- Validates scope isolation across multiple files

### 4. Data Validation and Conversion Test
- Tests data type conversion for Neo4j compatibility
- Verifies property validation and sanitization
- Tests error handling for invalid data
- Validates data transformation accuracy

### 5. Error Handling and Recovery Test
- Tests graceful handling of extraction errors
- Verifies database connection error recovery
- Tests data validation error handling
- Validates error reporting and logging

### 6. Performance and Scalability Test
- Tests processing of large codebases
- Verifies memory usage and performance
- Tests concurrent processing capabilities
- Validates scalability with increasing data size

### 7. Real-World Scenario Test
- Tests with realistic MATLAB codebases
- Verifies complex relationship patterns
- Tests edge cases and unusual code structures
- Validates practical usability

## Running the Tests

### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/matlab_test/test_matlab_extractor.py -v --asyncio-mode=auto

# Run specific unit test
python -m pytest tests/matlab_test/test_matlab_extractor.py::TestMatlabExtractor::test_scope_specific_variables -v
```

### Integration Tests
```bash
# Run all integration tests (requires Neo4j running)
python -m pytest tests/matlab_test/test_full_pipeline.py -v --asyncio-mode=auto

# Run specific integration test
python -m pytest tests/matlab_test/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_integration -v
```

### Full Test Suite
```bash
# Run both unit and integration tests
python -m pytest tests/matlab_test/ -v --asyncio-mode=auto

# Run with coverage
python -m pytest tests/matlab_test/ -v --asyncio-mode=auto --cov=neo4j_graphrag.experimental.components.code_extractor.matlab
```

## Test Data Requirements

The test data must demonstrate:

1. **Scope Isolation**: Variables with same names in different scopes create separate nodes
2. **Execution Order**: Script dependencies and execution order analysis
3. **Cross-Scope Access**: Proper handling of variable access between scopes
4. **Parameter Passing**: Distinction between parameter passing and cross-scope access
5. **Relationship Properties**: Detailed properties for relationships (line numbers, usage types, etc.)
6. **Complex Patterns**: Advanced MATLAB constructs and patterns
7. **Error Scenarios**: Invalid code and edge cases
8. **Performance Cases**: Large files and complex relationships

## Test Environment Setup

### Prerequisites
- Python 3.7+
- pytest
- pytest-asyncio
- pytest-cov (for coverage)
- Neo4j database (for integration tests)
- MATLAB Parser (included in the project)

### Neo4j Setup for Integration Tests
```bash
# Start Neo4j (Docker example)
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Or use local Neo4j installation
# Ensure Neo4j is running on bolt://localhost:7687
# Default credentials: neo4j/password
```

## Adding New Tests

### Unit Tests
1. Add new test methods to `TestMatlabExtractor` class
2. Follow the naming convention `test_<feature_name>`
3. Use async test methods for extractor operations
4. Include comprehensive assertions and error messages
5. Add test data files if needed

### Integration Tests
1. Add new test methods to `TestFullPipeline` class
2. Test complete pipeline workflows
3. Include database operations and validation
4. Test error scenarios and edge cases
5. Validate data integrity and consistency

### Test Data
1. Add new test files to appropriate directories
2. Ensure files demonstrate specific features or patterns
3. Update test scripts to include new files
4. Document the purpose and expected behavior

## Debugging

### Unit Test Debugging
1. Check test data files exist and are correctly formatted
2. Verify extractor configuration and parameters
3. Check for specific error messages in test output
4. Validate schema compliance and requirements
5. Test individual components in isolation

### Integration Test Debugging
1. Verify Neo4j database connection and configuration
2. Check database schema and constraints
3. Validate data transformation and conversion
4. Test pipeline components individually
5. Check error logs and database state

### Common Issues
1. **Scope Isolation**: Ensure variables have unique IDs across scopes
2. **Relationship Properties**: Verify all required properties are present
3. **Data Types**: Check Neo4j compatibility of property values
4. **Cross-File Processing**: Validate file dependency resolution
5. **Database Connection**: Ensure Neo4j is running and accessible

## Performance Considerations

### Unit Tests
- Keep test execution time under 30 seconds
- Use efficient data structures and algorithms
- Minimize I/O operations in tests
- Use mocking for external dependencies

### Integration Tests
- Test with realistic data sizes
- Monitor memory usage and performance
- Use database connection pooling
- Implement proper cleanup and teardown

## Coverage Requirements

### Code Coverage
- Aim for >90% code coverage
- Cover all major code paths and branches
- Test error handling and edge cases
- Include integration test coverage

### Feature Coverage
- All node types and properties
- All relationship types and patterns
- All scope scenarios and isolation
- All execution order analysis features
- All cross-file processing capabilities

## Continuous Integration

### Automated Testing
- Run unit tests on every commit
- Run integration tests on pull requests
- Generate coverage reports
- Validate schema compliance
- Test with multiple Python versions

### Quality Gates
- All tests must pass
- Coverage must meet minimum thresholds
- No critical security vulnerabilities
- Performance benchmarks must be met
- Documentation must be up to date

## Key Features Tested

- **Scope-Specific Variables**: Each variable gets a unique ID based on its scope
- **Execution Order Analysis**: Automatic detection of script dependencies
- **Cross-Scope Relationships**: Proper handling of variable access between scopes
- **Parameter Passing**: Correct distinction from cross-scope access
- **Relationship Properties**: Detailed metadata for all relationships
- **Schema Compliance**: Full adherence to the defined schema requirements
- **Data Persistence**: Reliable storage and retrieval from Neo4j
- **Error Handling**: Graceful handling of errors and edge cases
- **Performance**: Efficient processing of large codebases
- **Scalability**: Ability to handle complex and extensive MATLAB projects

## Notes

- The test files are designed to be self-documenting and demonstrate common MATLAB patterns
- Both unit and integration tests include detailed error messages to help identify issues
- The test data covers a variety of programming constructs to ensure comprehensive testing
- Scope isolation is a critical feature that must be properly tested
- Execution order analysis is essential for correct cross-scope relationship creation
- Integration tests require a running Neo4j database and may take longer to execute
- Performance tests help ensure the system can handle real-world workloads
- Error handling tests validate robustness and reliability of the system
