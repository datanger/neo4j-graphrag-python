#!/usr/bin/env python3
"""
简单的纯Python实现测试
"""

import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.append(str(Path(__file__).parent))

from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import (
    MatlabExtractor, _global_registry
)
from neo4j_graphrag.experimental.components.types import (
    TextChunk, TextChunks, DocumentInfo, LexicalGraphConfig
)

# 使用与主程序相同的schema和examples
from examples.customize.build_graph.pipeline.kg_builder_from_code import SCHEMA, EXAMPLES

async def test_pure_python_simple():
    """测试纯Python实现的简单案例"""
    print("=== 测试纯Python实现的简单案例 ===")
    
    # 创建测试目录和文件
    test_dir = Path("test_pure_python_simple")
    test_dir.mkdir(exist_ok=True)
    
    # 创建启动脚本
    entry_script_content = """% 启动脚本 - main.m
config_file = 'config.json';
data_path = '/data/input';
setup_config;
process_data;
"""
    
    entry_script_path = test_dir / "main.m"
    with open(entry_script_path, 'w') as f:
        f.write(entry_script_content)
    
    # 创建其他脚本
    scripts = {
        "setup_config.m": """% 设置配置脚本
function setup_config()
    global config_file;
    disp('Setting up configuration...');
    config = loadjson(config_file);
    workspace_path = '/workspace';
end
""",
        "process_data.m": """% 数据处理脚本
function process_data()
    global data_path;
    disp('Processing data...');
    input_file = data_path;
    raw_data = load(input_file);
    processed_data = raw_data * 2;
end
"""
    }
    
    for filename, content in scripts.items():
        with open(test_dir / filename, 'w') as f:
            f.write(content)
    
    print(f"创建测试文件在: {test_dir}")
    
    # 测试：使用纯Python实现（不传入llm参数）
    print("\n--- 测试纯Python实现 ---")
    MatlabExtractor.reset_global_registry()
    
    # 注意：不传入llm参数，默认使用纯Python实现
    extractor = MatlabExtractor(entry_script_path=str(entry_script_path))
    
    # 处理所有文件
    all_results = []
    for file_path in test_dir.glob("*.m"):
        print(f"处理文件: {file_path.name}")
        with open(file_path, 'r') as f:
            content = f.read()
        
        chunk = TextChunk(
            text=content,
            index=0,
            metadata={"file_path": str(file_path), "file_name": file_path.name, "code_type": "matlab"}
        )
        
        doc_info = DocumentInfo(
            path=str(file_path),
            metadata={"name": file_path.name}
        )
        
        result = await extractor.run(
            chunks=TextChunks(chunks=[chunk]),
            schema=SCHEMA,
            document_info=doc_info,
            lexical_graph_config=LexicalGraphConfig(),
            examples=EXAMPLES,
            enable_post_processing=True,
        )
        all_results.append(result)
    
    # 合并所有结果
    if all_results:
        final_result = all_results[0]
        for result in all_results[1:]:
            final_result.graph.nodes.extend(result.graph.nodes)
            final_result.graph.relationships.extend(result.graph.relationships)
    else:
        final_result = None
    
    if final_result:
        print(f"纯Python实现的结果: {len(final_result.graph.nodes)} 节点, {len(final_result.graph.relationships)} 关系")
        
        # 显示节点
        print("\n节点:")
        for node in final_result.graph.nodes:
            print(f"  - {node.id}: {node.label} ({node.properties.get('name', 'N/A')})")
        
        # 显示关系
        print("\n关系:")
        for rel in final_result.graph.relationships:
            print(f"  - {rel.start_node_id} -[{rel.type}]-> {rel.end_node_id}")
            if rel.properties:
                print(f"    属性: {rel.properties}")
        
        # 检查跨作用域关系
        print("\n--- 跨作用域关系分析 ---")
        
        # 查找USES关系
        uses_relations = [rel for rel in final_result.graph.relationships if rel.type == "USES"]
        print(f"找到 {len(uses_relations)} 个USES关系:")
        for rel in uses_relations:
            if "cross_scope" in rel.properties.get("usage_type", ""):
                print(f"  - 跨作用域USES: {rel.start_node_id} -> {rel.end_node_id}")
                print(f"    变量: {rel.properties.get('variable_name', 'N/A')}")
                print(f"    来源脚本: {rel.properties.get('source_script', 'N/A')}")
                print(f"    目标脚本: {rel.properties.get('target_script', 'N/A')}")
        
        # 查找ASSIGNED_TO关系
        assigned_to_relations = [rel for rel in final_result.graph.relationships if rel.type == "ASSIGNED_TO"]
        print(f"\n找到 {len(assigned_to_relations)} 个ASSIGNED_TO关系:")
        for rel in assigned_to_relations:
            if "cross_scope" in rel.properties.get("type", ""):
                print(f"  - 跨作用域ASSIGNED_TO: {rel.start_node_id} -> {rel.end_node_id}")
                print(f"    变量: {rel.properties.get('variable_name', 'N/A')}")
                print(f"    来源脚本: {rel.properties.get('source_script', 'N/A')}")
                print(f"    目标脚本: {rel.properties.get('target_script', 'N/A')}")
    else:
        print("没有生成任何结果")
    
    # 显示启动脚本分析的信息
    print(f"\n启动脚本执行流程: {_global_registry.entry_script_execution_flow}")
    print(f"启动脚本变量作用域: {_global_registry.entry_script_variable_scope}")
    print(f"执行顺序: {_global_registry.execution_order}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n清理测试文件: {test_dir}")

if __name__ == "__main__":
    asyncio.run(test_pure_python_simple())
