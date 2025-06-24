#!/usr/bin/env python3
"""
详细测试启动脚本驱动的MATLAB跨作用域关系处理器
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neo4j_graphrag.experimental.components.code_extractor.matlab.entry_script_driven_processor import EntryScriptDrivenProcessor
from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode

def create_test_files():
    """创建测试文件"""
    test_dir = Path("test_detailed_data")
    test_dir.mkdir(exist_ok=True)

    # 启动脚本
    entry_content = """% 启动脚本
global_var = 10;
config_param = 'test_config';

% 调用脚本
script_a;
script_b;

% 使用脚本b中的变量
result = processed_data + global_var;
disp(['Final result: ' num2str(result)]);"""

    # 脚本A
    script_a_content = """% 脚本A
input_data = [1, 2, 3, 4, 5];
temp_var = input_data * 2;

% 使用启动脚本中的变量
adjusted_data = input_data + global_var;

% 调用辅助函数
processed_data = helper_function(adjusted_data);"""

    # 脚本B
    script_b_content = """% 脚本B
% 使用脚本A中的变量
if exist('processed_data', 'var')
    final_result = processed_data * 1.5;
else
    final_result = 0;
end

% 定义新变量供后续脚本使用
output_data = final_result;"""

    # 辅助函数
    helper_function_content = """function result = helper_function(input_data)
% 辅助函数
result = sum(input_data) + config_param;
end"""

    (test_dir / "main_script.m").write_text(entry_content, encoding='utf-8')
    (test_dir / "script_a.m").write_text(script_a_content, encoding='utf-8')
    (test_dir / "script_b.m").write_text(script_b_content, encoding='utf-8')
    (test_dir / "helper_function.m").write_text(helper_function_content, encoding='utf-8')

    return test_dir

async def test_detailed_processor():
    """详细测试处理器"""
    print("=== 详细测试启动脚本驱动的处理器 ===\n")

    # 创建测试文件
    test_dir = create_test_files()
    entry_script_path = str(test_dir / "main_script.m")

    # 初始化处理器
    processor = EntryScriptDrivenProcessor()
    processor.set_entry_script(entry_script_path)

    print(f"1. 启动脚本分析:")
    print(f"   启动脚本: {entry_script_path}")
    print(f"   执行顺序: {processor.script_execution_order}")
    print(f"   启动脚本变量: {list(processor.variable_scope_timeline.keys())}")

    # 注册脚本变量
    for file_path in test_dir.glob("*.m"):
        if file_path.name != "main_script.m":
            script_name = file_path.stem
            content = file_path.read_text(encoding='utf-8')
            processor.register_script_variables(script_name, content)

    # 注册启动脚本的变量使用
    entry_content = (test_dir / "main_script.m").read_text(encoding='utf-8')
    processor.register_script_variables("entry_script", entry_content)

    print(f"\n2. 脚本变量注册:")
    print(f"   脚本A变量定义: {list(processor.variable_definitions.get('script_a', {}).keys())}")
    print(f"   脚本B变量定义: {list(processor.variable_definitions.get('script_b', {}).keys())}")
    print(f"   脚本A变量使用: {list(processor.variable_usage.get('script_a', {}).keys())}")
    print(f"   脚本B变量使用: {list(processor.variable_usage.get('script_b', {}).keys())}")

    # 创建测试图
    nodes = [
        Neo4jNode(id="script_main_script_test", label="Script",
                 properties={'name': 'main_script', 'code_snippet': 'test'}),
        Neo4jNode(id="script_script_a_test", label="Script",
                 properties={'name': 'script_a', 'code_snippet': 'test'}),
        Neo4jNode(id="script_script_b_test", label="Script",
                 properties={'name': 'script_b', 'code_snippet': 'test'}),
        Neo4jNode(id="function_helper_function_test", label="Function",
                 properties={'name': 'helper_function', 'code_snippet': 'test'})
    ]

    test_graph = Neo4jGraph(nodes=nodes, relationships=[])

    print(f"\n3. 处理跨作用域关系...")
    final_graph = processor.process_cross_scope_relationships(test_graph)

    print(f"\n4. 结果分析:")
    print(f"   最终节点数: {len(final_graph.nodes)}")
    print(f"   最终关系数: {len(final_graph.relationships)}")

    # 分析跨作用域关系
    cross_scope_rels = [rel for rel in final_graph.relationships
                       if rel.properties.get('post_processed', False)]

    print(f"   跨作用域关系数: {len(cross_scope_rels)}")

    # 按类型统计关系
    rel_types = {}
    for rel in cross_scope_rels:
        rel_type = rel.type
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

    print(f"   关系类型分布:")
    for rel_type, count in rel_types.items():
        print(f"     {rel_type}: {count}")

    # 显示具体的跨作用域关系
    print(f"\n5. 具体的跨作用域关系:")
    for i, rel in enumerate(cross_scope_rels, 1):
        print(f"   {i}. {rel.start_node_id} -[{rel.type}]-> {rel.end_node_id}")
        print(f"      属性: {rel.properties}")

    # 验证关键关系
    print(f"\n6. 验证关键跨作用域关系:")

    # 检查脚本A使用启动脚本中的global_var
    global_var_uses = [rel for rel in cross_scope_rels
                      if rel.type == 'USES' and
                      'global_var' in rel.properties.get('variable_name', '')]

    if global_var_uses:
        print(f"   ✓ 找到脚本A使用启动脚本中global_var的关系")
    else:
        print(f"   ✗ 未找到脚本A使用启动脚本中global_var的关系")

    # 检查脚本B使用脚本A中的processed_data
    processed_data_uses = [rel for rel in cross_scope_rels
                          if rel.type == 'USES' and
                          'processed_data' in rel.properties.get('variable_name', '')]

    if processed_data_uses:
        print(f"   ✓ 找到脚本B使用脚本A中processed_data的关系")
    else:
        print(f"   ✗ 未找到脚本B使用脚本A中processed_data的关系")

    # 检查启动脚本使用脚本B中的变量
    result_uses = [rel for rel in cross_scope_rels
                  if rel.type == 'USES' and
                  'processed_data' in rel.properties.get('variable_name', '') and
                  'entry_script' in rel.start_node_id]

    if result_uses:
        print(f"   ✓ 找到启动脚本使用脚本A中processed_data的关系")
    else:
        print(f"   ✗ 未找到启动脚本使用脚本A中processed_data的关系")

    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_detailed_processor())
