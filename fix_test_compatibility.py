#!/usr/bin/env python3
"""
修复测试文件中的对象访问方式
将Neo4jNode和Neo4jRelationship的字典访问方式改为属性访问
"""

import re
from pathlib import Path

def fix_test_file(file_path):
    """修复测试文件中的对象访问方式"""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复节点访问方式
    # node.get('type') -> node.label
    # node.get('id', '') -> node.id
    # node.get('properties', {}) -> node.properties
    # node.get('properties', {}).get('name', '') -> node.properties.get('name', '')
    
    # 修复关系访问方式
    # rel.get('type') -> rel.type
    # rel.get('source_id', '') -> rel.start_node_id
    # rel.get('target_id', '') -> rel.end_node_id
    # rel.get('source_type', '') -> rel.start_node_type
    # rel.get('target_type', '') -> rel.end_node_type
    
    # 修复节点类型检查
    content = re.sub(r'node\.get\(\'type\'\) == \'(\w+)\'', r'node.label == \'\1\'', content)
    content = re.sub(r'node\.get\(\'type\'\)', r'node.label', content)
    
    # 修复节点ID访问
    content = re.sub(r'node\.get\(\'id\', \'\'\)', r'node.id', content)
    content = re.sub(r'node\.get\(\'id\'\)', r'node.id', content)
    
    # 修复节点属性访问
    content = re.sub(r'node\.get\(\'properties\', \{\}\)', r'node.properties', content)
    content = re.sub(r'node\.get\(\'properties\'\)', r'node.properties', content)
    
    # 修复关系类型访问
    content = re.sub(r'rel\.get\(\'type\'\)', r'rel.type', content)
    
    # 修复关系节点ID访问
    content = re.sub(r'rel\.get\(\'source_id\', \'\'\)', r'rel.start_node_id', content)
    content = re.sub(r'rel\.get\(\'source_id\'\)', r'rel.start_node_id', content)
    content = re.sub(r'rel\.get\(\'target_id\', \'\'\)', r'rel.end_node_id', content)
    content = re.sub(r'rel\.get\(\'target_id\'\)', r'rel.end_node_id', content)
    
    # 修复关系节点类型访问
    content = re.sub(r'rel\.get\(\'source_type\', \'\'\)', r'rel.start_node_type', content)
    content = re.sub(r'rel\.get\(\'source_type\'\)', r'rel.start_node_type', content)
    content = re.sub(r'rel\.get\(\'target_type\', \'\'\)', r'rel.end_node_type', content)
    content = re.sub(r'rel\.get\(\'target_type\'\)', r'rel.end_node_type', content)
    
    # 修复嵌套属性访问
    content = re.sub(r'var\.get\(\'properties\', \{\}\)\.get\(\'name\', \'\'\)', r'var.properties.get(\'name\', \'\')', content)
    content = re.sub(r'var\.get\(\'properties\', \{\}\)\.get\(\'scope_id\', \'\'\)', r'var.properties.get(\'scope_id\', \'\')', content)
    content = re.sub(r'var\.get\(\'properties\', \{\}\)\.get\(\'scope_type\', \'\'\)', r'var.properties.get(\'scope_type\', \'\')', content)
    
    # 修复函数节点属性访问
    content = re.sub(r'func_node\.get\(\'properties\', \{\}\)\.get\(\'parameters\', \'\'\)', r'func_node.properties.get(\'parameters\', \'\')', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {file_path}")

def main():
    """主函数"""
    test_files = [
        "tests/matlab_test/test_comprehensive.py",
        "tests/matlab_test/test_all.py",
        "tests/matlab_test/test_updated_requirements.py"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            fix_test_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main() 