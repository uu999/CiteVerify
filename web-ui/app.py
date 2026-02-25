# -*- coding: utf-8 -*-
"""
CiteVerify Web UI

基于 Flask 的简单 Web 界面
"""
import os
import sys
import json
import logging
import threading
from flask import Flask, render_template, request, jsonify, Response
from queue import Queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
# 设置 citeverify 模块的日志级别
logging.getLogger('citeverify').setLevel(logging.INFO)

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citeverify.full_pipeline import FullPipeline, run_full_pipeline
from citeverify.models import YaYiDocParserConfig

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

# 全局进度存储
progress_store = {}


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    分析论文 API
    
    请求参数：
    - pdf_url: PDF 链接
    - citation_format: 引用格式 (numeric/author_year)
    - reference_format: 参考文献格式 (apa/ieee/mla/gb_t_7714/chicago/harvard/vancouver/other)
    - llm_model: LLM 模型名称
    - llm_api_key: LLM API Key
    - llm_base_url: LLM API Base URL (可选)
    - semantic_scholar_key: Semantic Scholar API Key (可选)
    """
    try:
        data = request.json
        
        pdf_url = data.get('pdf_url', '').strip()
        citation_format = data.get('citation_format', 'numeric')
        reference_format = data.get('reference_format', 'apa')
        llm_model = data.get('llm_model', 'gpt-4o-mini')
        llm_api_key = data.get('llm_api_key', '').strip()
        llm_base_url = data.get('llm_base_url', '').strip() or None
        semantic_scholar_key = data.get('semantic_scholar_key', '').strip() or None
        
        if not pdf_url:
            return jsonify({'success': False, 'error': '请输入论文 PDF 链接'})
        
        # 参考文献格式为「其他」时，必须配置 LLM
        if reference_format == 'other':
            if not llm_api_key:
                return jsonify({'success': False, 'error': '选择「其他」格式时需配置 LLM API Key，将整段参考文献交给 LLM 提取条目'})
            if not llm_model:
                return jsonify({'success': False, 'error': '选择「其他」格式时需配置 LLM 模型名称'})
        
        # 转换参数
        listing_style = 'numbered' if citation_format == 'numeric' else 'author_year'
        
        # 创建任务 ID
        import uuid
        task_id = str(uuid.uuid4())
        progress_store[task_id] = {
            'step': 'starting',
            'message': '正在启动...',
            'progress': 0,
            'completed': False,
            'result': None,
        }
        
        # 在后台线程运行
        def run_task():
            logger = logging.getLogger('citeverify')
            logger.info("[Analyze] 分析任务开始 task_id=%s pdf_url=%s", task_id[:8], pdf_url[:60] + "..." if len(pdf_url) > 60 else pdf_url)
            def progress_callback(step, message, progress):
                progress_store[task_id].update({
                    'step': step,
                    'message': message,
                    'progress': progress,
                })
            
            try:
                result = run_full_pipeline(
                    pdf_url=pdf_url,
                    citation_format=reference_format,
                    listing_style=listing_style,
                    llm_model_name=llm_model,
                    llm_api_key=llm_api_key if llm_api_key else None,
                    llm_base_url=llm_base_url,
                    semantic_scholar_api_key=semantic_scholar_key,
                    progress_callback=progress_callback,
                )
                
                progress_store[task_id].update({
                    'completed': True,
                    'result': result.to_dict(),
                })
                logger.info("[Analyze] 分析任务结束 task_id=%s success=True", task_id[:8])
            except Exception as e:
                import traceback
                traceback.print_exc()
                progress_store[task_id].update({
                    'completed': True,
                    'result': {
                        'success': False,
                        'error': str(e),
                    }
                })
                logger.info("[Analyze] 分析任务结束 task_id=%s success=False error=%s", task_id[:8], str(e))
        
        thread = threading.Thread(target=run_task)
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '任务已提交',
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    """获取任务进度"""
    if task_id not in progress_store:
        return jsonify({'success': False, 'error': '任务不存在'})
    
    return jsonify({
        'success': True,
        **progress_store[task_id],
    })


@app.route('/api/result/<task_id>')
def get_result(task_id):
    """获取任务结果"""
    if task_id not in progress_store:
        return jsonify({'success': False, 'error': '任务不存在'})
    
    task = progress_store[task_id]
    if not task['completed']:
        return jsonify({'success': False, 'error': '任务尚未完成'})
    
    return jsonify({
        'success': True,
        'result': task['result'],
    })


if __name__ == '__main__':
    # 确保模板目录存在
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    print("=" * 60)
    print("CiteVerify Web UI")
    print("=" * 60)
    print("访问地址: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
