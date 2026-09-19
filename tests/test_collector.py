"""Mock SSE and gate tests: no network, CUDA, GPU or model invocation."""
import io
import json
import unittest

from bench.collector import (CollectorError, assert_local_endpoint, build_request,
                             collect_stream, enforce_final_gate)


def sse(*chunks):
    lines = [b'data: ' + json.dumps(item).encode('utf-8') + b'\n\n'
             for item in chunks]
    return io.BytesIO(b''.join(lines) + b'data: [DONE]\n\n')


class CollectorTests(unittest.TestCase):
    def test_endpoint_only_ipv4_loopback_exact_route(self):
        assert_local_endpoint('http://127.0.0.1:8000/v1/chat/completions')
        for url in ('http://localhost:8000/v1/chat/completions',
                    'https://127.0.0.1:8000/v1/chat/completions',
                    'http://127.0.0.1:8000/v1/chat/completions?x=1',
                    'http://127.0.0.1:8000@evil.invalid/v1/chat/completions',
                    'http://127.0.0.1:8000/v1/chat/completions#x'):
            with self.subTest(url=url), self.assertRaises((CollectorError, ValueError)):
                assert_local_endpoint(url)

    def test_live_gate_cannot_be_satisfied_by_flag_only(self):
        armed = {'status': 'static-contract-armed', 'inference_or_gpu_testing': 'NOT PERFORMED'}
        enforce_final_gate(execute=False, authorization=None, armed=None)
        with self.assertRaises(CollectorError):
            enforce_final_gate(execute=True, authorization=None, armed=armed)
        with self.assertRaises(CollectorError):
            enforce_final_gate(execute=True, authorization='I_AUTHORIZE_FINAL_GPU_BENCHMARKS', armed=None)
        enforce_final_gate(execute=True, authorization='I_AUTHORIZE_FINAL_GPU_BENCHMARKS', armed=armed)

    def test_complete_stream_uses_real_usage_not_guess(self):
        stream = sse({'choices':[{'delta':{'content':'42'},'finish_reason':None}]},
                     {'choices':[{'delta':{},'finish_reason':'stop'}]},
                     {'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':2}})
        result = collect_stream(stream)
        self.assertEqual(result['response'], '42')
        self.assertEqual(result['metrics']['output_tokens'], 2)
        self.assertEqual(result['metrics']['input_tokens'], 10)
        self.assertGreaterEqual(result['metrics']['e2e_ms'], result['metrics']['ttft_ms'])

    def test_incomplete_sse_and_missing_usage_rejected(self):
        no_usage = io.BytesIO(b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n')
        with self.assertRaisesRegex(CollectorError, 'Incomplete'):
            collect_stream(no_usage)
        with self.assertRaisesRegex(CollectorError, 'Incomplete'):
            collect_stream(io.BytesIO(b'data: [DONE]\n'))

    def test_streamed_tool_arguments_reassembled_and_validated(self):
        stream = sse({'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':'file_', 'arguments':'{"filename":'}}]}}]},
                     {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':'search','arguments':'"main.py"}'}}]},'finish_reason':'tool_calls'}]},
                     {'choices':[],'usage':{'prompt_tokens':20,'completion_tokens':6}})
        result = collect_stream(stream)
        self.assertEqual(json.loads(result['response']),
                         {'name':'file_search','arguments':{'filename':'main.py'}})

    def test_tool_payload_contains_real_function_schema(self):
        case = {'kind':'tool_call','prompt':'Read a file',
                'tool':{'name':'file_search','arguments':{'filename':'main.py'}}}
        body = build_request(case,'local-model',64)
        self.assertEqual(body['tools'][0]['function']['parameters']['additionalProperties'], False)
        self.assertEqual(body['stream_options'], {'include_usage':True})
