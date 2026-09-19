"""Trusted fixture self-tests, no model-generated code or GPU execution."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from bench.agent_replay import replay
from bench.agent_loop import ScriptedTransport, run_session

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'bench/agent_tasks/config_merge_v1'
PRIVATE = ROOT / '.local/benchmark/config-agent-unit'


class ConfigAgentTaskTests(unittest.TestCase):
    def test_reference_resolves_all_sealed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            shutil.copytree(TASK / 'workspace', target/'workspace')
            shutil.copytree(TASK / 'grader', target/'grader')
            oracle = (TASK/'grader/reference_config.py').read_text(encoding='utf-8')
            (target/'workspace/src/config.py').write_text(oracle,encoding='utf-8')
            for folder, n in [('tests',2),('../grader',5)]:
                result = subprocess.run([sys.executable,'-B','-m','unittest','discover',
                                         '-s',folder,'-q'],cwd=target/'workspace',
                                         text=True,capture_output=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn(f'Ran {n} tests',result.stderr)

    def test_scripted_agent_receives_only_workspace_content(self):
        source = (TASK/'workspace/src/config.py').read_text(encoding='utf-8')
        calls = [{'tool':'read_file','arguments':{'path':'README.md'}},
                 {'tool':'read_file','arguments':{'path':'src/config.py'}},
                 {'tool':'write_file','arguments':{'path':'src/config.py','content':source+'\n# mock edit\n'}},
                 {'tool':'run_tests','arguments':{}},
                 {'tool':'finish','arguments':{'summary':'Ready for isolated verification'}}]
        transport=ScriptedTransport(calls)
        receipt=run_session(TASK,transport,PRIVATE)
        self.assertEqual(receipt['status'],'unverified')
        self.assertNotIn('reference_config',str(transport.seen_messages))
        self.assertNotIn('test_private',str(transport.seen_messages))
