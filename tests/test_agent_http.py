"""Scripted HTTP integration: never imports a model, invokes Docker, or uses a GPU."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch
from bench.agent_http import OpenAITransport, encode_messages, run_live
from bench.agent_loop import run_session
from bench.agent_replay import AgentTaskError
from tests.test_agent_replay import TASK, PRIVATE, transcript
from bench.repo_judge import grade_visible, RepoJudgeError


class ScriptedAPI(BaseHTTPRequestHandler):
    calls = []
    requests = []
    def do_POST(self):
        size = int(self.headers['Content-Length'])
        payload = json.loads(self.rfile.read(size))
        self.requests.append(payload)
        if not self.calls:
            self.send_error(409)
            return
        action = self.calls.pop(0)
        response = {'choices': [{'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': None, 'tool_calls': [
                    {'type': 'function', 'id': 'call_script', 'function': {
                        'name': action['tool'], 'arguments': json.dumps(action['arguments'])}}]}}],
                    'usage': {'prompt_tokens': 12, 'completion_tokens': 9}}
        raw = json.dumps(response).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def log_message(self, *args):
        pass


class HttpAgentTests(unittest.TestCase):
    def test_real_loopback_scripted_tool_session(self):
        ScriptedAPI.calls = transcript()
        ScriptedAPI.requests = []
        server = HTTPServer(('127.0.0.1', 0), ScriptedAPI)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
            transport = OpenAITransport(url, 'offline-scripted-endpoint')
            result = run_session(TASK, transport, PRIVATE)
            self.assertEqual(result['turns'], 6)
            self.assertEqual(len(transport.usage), 6)
            self.assertEqual(len(ScriptedAPI.requests), 6)
            for request in ScriptedAPI.requests:
                self.assertEqual(request['tool_choice'], 'required')
                self.assertEqual(len(request['tools']), 5)
                self.assertNotIn('test_private', json.dumps(request))
            third = ScriptedAPI.requests[2]['messages']
            assistant = [m for m in third if m['role'] == 'assistant']
            tool = [m for m in third if m['role'] == 'tool']
            self.assertEqual(assistant[-1]['tool_calls'][0]['id'], tool[-1]['tool_call_id'])
            self.assertFalse(result['generated_code_executed'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)

    def test_transport_rejects_missing_usage_or_prose(self):
        t = OpenAITransport('http://127.0.0.1:9000/v1/chat/completions', 'mock',
                            sender=lambda body: {'choices': [{'finish_reason':'stop',
                                                    'message': {'content': 'looks good'}}]})
        with self.assertRaises(AgentTaskError):
            t.complete([{'role': 'user', 'content': 'hello'}])
        t.sender = lambda body: {'choices':[{'finish_reason':'tool_calls',
                   'message':{'tool_calls':[{'id':'id','type':'function',
                   'function':{'name':'finish','arguments':'{"summary":"done"}'}}]}}]}
        with self.assertRaises(AgentTaskError):
            t.complete([{'role': 'user', 'content': 'hello'}])

    def test_visible_feedback_is_isolated_from_hidden_grader(self):
        with patch('bench.repo_judge.subprocess.run') as invoke:
            import subprocess
            invoke.return_value = subprocess.CompletedProcess('docker', 0, '', 'Ran 2 tests in 0.001s\nOK')
            receipt = grade_visible(TASK, TASK/'workspace',
                 image='python:3.12-slim@sha256:' + 'a'*64,
                 private_root=PRIVATE, authorized=True)
        self.assertEqual(receipt['status'], 'passed')
        self.assertEqual(receipt['tests_ran'], 2)
        self.assertNotIn('test_private', str(invoke.call_args))
        self.assertNotIn('test_private', str(receipt))
        self.assertIn('--network=none', invoke.call_args.args[0])
        with self.assertRaises(RepoJudgeError):
            grade_visible(TASK, TASK/'workspace',
                  image='python:3.12-slim@sha256:' + 'a'*64,
                  private_root=PRIVATE)

    def test_live_agent_never_runs_before_explicit_gate(self):
        transport = OpenAITransport('http://127.0.0.1:9000/v1/chat/completions', 'mock',
                                    sender=lambda _: self.fail('Never contact model'))
        with self.assertRaises(AgentTaskError):
            run_live(TASK, transport, PRIVATE, image=None, allow_execution=False)
        self.assertEqual(transport.usage, [])


    def test_mocked_visible_pass_private_failure_is_task_failure(self):
        """All responses are scripted and both container receipts are mocked."""
        import subprocess
        actions = transcript()
        def sender(body):
            action = actions.pop(0)
            return {'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
                {'id':'script','type':'function','function':{
                    'name':action['tool'],'arguments':json.dumps(action['arguments'])}}]}}],
                'usage':{'prompt_tokens':12,'completion_tokens':9}}
        public = subprocess.CompletedProcess('docker', 0, '', 'Ran 2 tests in 0.001s\nOK')
        hidden = subprocess.CompletedProcess('docker', 1, '',
                        'Ran 2 tests in 0.001s\nOK\nRan 5 tests in 0.001s\nFAILED')
        transport = OpenAITransport('http://127.0.0.1:9000/v1/chat/completions',
                                    'scripted-only', sender=sender)
        with patch('bench.repo_judge.subprocess.run', side_effect=[public, hidden]) as docker:
            result = run_live(TASK, transport, PRIVATE,
                  image='python:3.12-slim@sha256:'+'a'*64, allow_execution=True)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['model_turns'], 6)
        self.assertEqual(result['model_completion_tokens'], 54)
        self.assertFalse(result['repo_receipt']['passed'])
        self.assertEqual(docker.call_count, 2)
        self.assertNotIn('test_private', str(docker.call_args_list[0]))
        self.assertIn('test_private', str(docker.call_args_list[1]))


if __name__ == '__main__':
    unittest.main()
