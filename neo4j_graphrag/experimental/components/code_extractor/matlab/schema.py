from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, PropertyType, RelationshipType

# Schema definitions for MATLAB code analysis
SCHEMA = GraphSchema(
    node_types=[
        NodeType(
            label="Function",
            description="A code function definition",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the function"),
                PropertyType(name="file_path", type="STRING", description="Path to the file containing the function"),
                PropertyType(name="line_range", type="STRING", description="Line range where the function is defined"),
                PropertyType(name="description", type="STRING", description="Function description from docstring"),
                PropertyType(name="parameters", type="STRING", description="List of function parameters"),
                PropertyType(name="returns", type="STRING", description="List of return values"),
            ],
        ),
        NodeType(
            label="Variable",
            description="A variable used in the code",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the variable"),
                PropertyType(name="file_path", type="STRING", description="Path to the file where the variable is defined"),
                PropertyType(name="scope_id", type="STRING", description="ID of the scope (script or function) where this variable is defined"),
                PropertyType(name="scope_type", type="STRING", description="Type of scope: 'script' or 'function'"),
                PropertyType(name="line_range", type="LIST", description="List of tuples containing variable usage in script and corresponding line range, each tuple element is like (context, start_line-end_line)"),
            ],
        ),
        NodeType(
            label="Script",
            description="A code script file",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the script"),
                PropertyType(name="file_path", type="STRING", description="Path to the script file"),
                PropertyType(name="description", type="STRING", description="Script description"),
            ],
        ),
    ],
    relationship_types=[
        RelationshipType(
            label="CALLS",
            description="A function or script calls another function or script"
        ),
        RelationshipType(
            label="USES",
            description="A function or script uses a variable which is defined in another function or script",
        ),
        RelationshipType(
            label="DEFINES",
            description="A script defines a variable or a function and a function defines a variable",
        ),
        RelationshipType(
            label="ASSIGNED_TO",
            description="A variable is assigned to another variable",
        ),
        RelationshipType(
            label="MODIFIES",
            description="A function or script modifies a variable that was defined in another scope",
        ),
    ],
    patterns=[
        ("Function", "CALLS", "Function"),
        ("Function", "CALLS", "Script"),
        ("Script", "CALLS", "Function"),
        ("Script", "CALLS", "Script"),
        ("Function", "USES", "Variable"),
        ("Script", "USES", "Variable"),
        ("Function", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Function"),
        ("Function", "MODIFIES", "Variable"),
        ("Script", "MODIFIES", "Variable"),
        ("Variable", "ASSIGNED_TO", "Variable"),
    ],
) 