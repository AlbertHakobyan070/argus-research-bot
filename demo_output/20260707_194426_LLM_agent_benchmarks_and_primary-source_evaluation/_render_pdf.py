import sys, os
sys.path.insert(0, r'A:\Hermes\Agents\intel-stack\scripts')
os.environ.pop('PYTHONPATH', None)
from _common import markdown_to_pdf
markdown_to_pdf(open(r'A:\Hermes\Agents\argus\demo_output\20260707_194426_LLM_agent_benchmarks_and_primary-source_evaluation\report.md','r',encoding='utf-8').read(),
                  r'A:\Hermes\Agents\argus\demo_output\20260707_194426_LLM_agent_benchmarks_and_primary-source_evaluation\report.pdf', title=r'LLM_agent_benchmarks_and_primary-source_evaluation')
