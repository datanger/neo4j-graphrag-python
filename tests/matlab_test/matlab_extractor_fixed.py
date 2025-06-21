"""MATLAB code extractor with fixed script and function call handling."""
import re
from typing import Dict, List, Set, Tuple, Optional, Any

# Constants
IDENTIFIERS_TO_EXCLUDE = {
    'if', 'else', 'elseif', 'end', 'for', 'while', 'function', 'return',
    'break', 'continue', 'switch', 'case', 'otherwise', 'try', 'catch',
    'classdef', 'properties', 'methods', 'events', 'enumeration', 'parfor',
    'spmd', 'persistent', 'global', 'import', 'arguments', 'function_handle'
}

class MatlabExtractor:
    """Extracts MATLAB code elements and their relationships."""
    
    def __init__(self, debug: bool = False):
        """Initialize the MATLAB extractor."""
        self.debug = debug
        self.nodes = []
        self.edges = []
        self.current_function = None
        self.current_scope = None
        self.variable_definitions = {}
        self.known_function_names = set()
        self.processed_nodes = set()
        
    def extract(self, file_path: str, file_content: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract nodes and edges from MATLAB code."""
        self.file_path = file_path
        self.text = file_content
        self._extract_functions()
        self._extract_script()
        self._extract_variables_and_relationships()
        return self.nodes, self.edges
    
    def _add_node(self, node: Dict) -> None:
        """Add a node if it doesn't already exist."""
        if not any(n.get('id') == node['id'] for n in self.nodes):
            self.nodes.append(node)
    
    def _add_edge(self, source_id: str, target_id: str, label: str, 
                 properties: Optional[Dict] = None) -> None:
        """Add an edge if it doesn't already exist."""
        if properties is None:
            properties = {}
            
        edge = {
            'source': source_id,
            'target': target_id,
            'label': label,
            'properties': properties
        }
        
        if edge not in self.edges:
            self.edges.append(edge)
    
    def _get_line_number(self, text: str, offset: int = 0) -> int:
        """Get the line number from text with optional offset."""
        return text.count('\n') + 1 + offset
    
    def _extract_functions(self) -> None:
        """Extract function definitions from the file."""
        pattern = r'^\s*function\s+(?:\[.*\]|\s*\w+\s*)(?:\s*=\s*|\s+)(\w+)'
        
        for match in re.finditer(pattern, self.text, re.MULTILINE):
            func_name = match.group(1)
            func_id = f"func_{func_name}_{len(self.nodes)}"
            
            # Add function node
            func_node = {
                'id': func_id,
                'label': 'Function',
                'name': func_name,
                'file_path': self.file_path,
                'line_range': self._get_line_range(match.start())
            }
            self._add_node(func_node)
            self.known_function_names.add(func_name.lower())
    
    def _extract_script(self) -> None:
        """Extract script information if no functions are found."""
        if not any(n['label'] == 'Function' for n in self.nodes):
            script_name = self.file_path.split('/')[-1].replace('.m', '')
            script_id = f"script_{script_name}_{len(self.nodes)}"
            
            script_node = {
                'id': script_id,
                'label': 'Script',
                'name': script_name,
                'file_path': self.file_path,
                'line_range': (1, self._get_line_number(self.text))
            }
            self._add_node(script_node)
    
    def _get_line_range(self, pos: int) -> Tuple[int, int]:
        """Get start and end line numbers for a code block."""
        start_line = self.text.count('\n', 0, pos) + 1
        end_line = self.text.count('\n', pos) + 1
        return (start_line, end_line)
    
    def _process_script_calls(self, code: str, caller_id: str, file_path: str, line_offset: int) -> None:
        """Process script calls (run, eval, etc.) in the given code."""
        script_call_patterns = [
            r'(?:run\s*\(\s*[\'\"]|eval\s*\(\s*[\'\"])([^\'\"]+\.m)[\'\"]\s*\)',
            r'(?<![\w.])([a-zA-Z_]\w*)(?=\s*\()'
        ]
        
        for pattern in script_call_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                script_name = match.group(1) if match.lastindex >= 1 else match.group(0)
                script_name = script_name.replace('.m', '').strip()
                
                if script_name.lower() in IDENTIFIERS_TO_EXCLUDE or \
                   script_name.lower() in self.known_function_names:
                    continue
                
                # Find the called script in our nodes
                called_script = next((n for n in self.nodes 
                                   if n.get('name', '').lower() == script_name.lower()
                                   and n.get('label') == 'Script'), None)
                
                if called_script:
                    match_line = self._get_line_number(code[:match.start()], line_offset) + 1
                    
                    # Check if this call relationship already exists
                    call_exists = any(
                        e.get('source') == caller_id and 
                        e.get('target') == called_script['id'] and 
                        e.get('label') == 'CALLS' and
                        e.get('properties', {}).get('line_number') == match_line
                        for e in self.edges
                    )
                    
                    if not call_exists:
                        self._add_edge(
                            source_id=caller_id,
                            target_id=called_script['id'],
                            label='CALLS',
                            properties={
                                'file': file_path,
                                'line_number': match_line,
                                'call_type': 'script_call'
                            }
                        )
    
    def _process_function_calls(self, code: str, caller_id: str, file_path: str, line_offset: int) -> None:
        """Process function calls in the given code."""
        if not hasattr(self, 'call_sites'):
            self.call_sites = {}
            
        code_hash = hash(code.strip())
        if (caller_id, code_hash) in self.processed_nodes:
            return
            
        self.processed_nodes.add((caller_id, code_hash))
        
        # Process script calls first
        self._process_script_calls(code, caller_id, file_path, line_offset)
        
        # Look for function calls
        func_call_pattern = r'(?<![\w.])([a-zA-Z_]\w*)\s*(?=\()'
        
        for match in re.finditer(func_call_pattern, code):
            func_name = match.group(1)
            
            if func_name.lower() in IDENTIFIERS_TO_EXCLUDE or \
               func_name.lower() in self.known_function_names:
                continue
                
            # Find the called function (case-insensitive)
            called_func = next((n for n in self.nodes 
                             if n.get('name', '').lower() == func_name.lower() 
                             and n.get('label') in ['Function', 'Script']), None)
            
            if not called_func:
                # Try to find by ID as fallback
                func_id = f"func_{func_name.lower()}_"
                called_func = next((n for n in self.nodes 
                                 if n.get('id', '').startswith(func_id)
                                 and n.get('label') in ['Function', 'Script']), None)
                if not called_func:
                    continue
            
            match_line = self._get_line_number(code[:match.start()], line_offset) + 1
            
            # Check if this call relationship already exists
            call_exists = any(
                e.get('source') == caller_id and 
                e.get('target') == called_func['id'] and 
                e.get('label') == 'CALLS' and
                e.get('properties', {}).get('line_number') == match_line
                for e in self.edges
            )
            
            if not call_exists:
                # Track call sites for debugging
                self.call_sites[(caller_id, called_func['id'])] = {
                    'caller': caller_id,
                    'callee': called_func['id'],
                    'line': match_line,
                    'file': file_path
                }
                
                # Add CALLS relationship
                self._add_edge(
                    source_id=caller_id,
                    target_id=called_func['id'],
                    label='CALLS',
                    properties={
                        'file': file_path,
                        'line_number': match_line,
                        'call_type': 'function_call'
                    }
                )
    
    def _extract_variables_and_relationships(self) -> None:
        """Extract variables and their relationships."""
        for node in self.nodes:
            if node['label'] in ['Function', 'Script']:
                self._process_node_variables(node)
    
    def _process_node_variables(self, node: Dict) -> None:
        """Process variables within a function or script node."""
        if 'line_range' not in node:
            return
            
        start_line, end_line = node['line_range']
        code_lines = self.text.split('\n')[start_line-1:end_line]
        code_block = '\n'.join(code_lines)
        
        # Process function calls and script calls
        self._process_function_calls(code_block, node['id'], node['file_path'], start_line - 1)
        
        # Process variable assignments
        assignment_pattern = r'(?<![=!<>~])=(?!=)'
        for match in re.finditer(assignment_pattern, code_block):
            line_start = code_block.rfind('\n', 0, match.start()) + 1
            var_part = code_block[line_start:match.start()].strip()
            
            # Skip if this is part of a comparison or other operation
            if any(op in var_part for op in ['==', '~=', '>', '<', '>=', '<=', '&', '|', '~']):
                continue
            
            # Get the variable name (last token before =)
            var_name = var_part.split()[-1] if var_part else ''
            
            if not var_name or not var_name.isidentifier() or var_name in IDENTIFIERS_TO_EXCLUDE:
                continue
            
            # Calculate line number
            line_num = code_block.count('\n', 0, match.start()) + start_line
            
            # Create or update variable node
            var_id = f"var_{var_name}_{len(self.nodes)}"
            var_node = {
                'id': var_id,
                'label': 'Variable',
                'name': var_name,
                'file_path': node['file_path'],
                'line_range': (line_num, line_num)
            }
            self._add_node(var_node)
            
            # Add DEFINES relationship
            self._add_edge(
                source_id=node['id'],
                target_id=var_id,
                label='DEFINES',
                properties={
                    'file': node['file_path'],
                    'line': line_num
                }
            )
