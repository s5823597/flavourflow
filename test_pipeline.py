"""Regression checks without network calls, credentials or a running Streamlit UI."""
import ast
import base64
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

class PipelineTests(unittest.TestCase):
    def test_frame_sampling_and_vision_request_limit(self):
        tree = ast.parse(Path(__file__).with_name('streamlit_app.py').read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_frames_and_analyse')
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Observed cooking action'))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        scope = dict(Path=Path, os=os, subprocess=subprocess, base64=base64,
                     GROQ_API_KEY='test-only', VISION_MODEL='qwen/qwen3.8-27b', Groq=lambda **kw: client)
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<pipeline>', 'exec'), scope)
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            try:
                os.chdir(tmp)
                subprocess.run(['ffmpeg','-loglevel','error','-f','lavfi','-i','testsrc2=size=128x72:rate=10','-t','2','-c:v','libx264','input.mp4'],check=True)
                desc, folder = scope['extract_frames_and_analyse']('input.mp4', 10)
                self.assertEqual(len(list(Path(folder).glob('frame_*.jpg'))), 10)
                self.assertEqual([len(c['messages'][0]['content'])-1 for c in calls], [3,3,3,1])
                self.assertEqual(desc.count('Observed cooking action'), 4)
                # A later run with fewer samples must not include stale frames.
                calls.clear()
                scope['extract_frames_and_analyse']('input.mp4', 5)
                self.assertEqual(len(list(Path(folder).glob('frame_*.jpg'))), 5)
                self.assertEqual([len(c['messages'][0]['content'])-1 for c in calls], [3,2])
            finally:
                os.chdir(old)

if __name__ == '__main__':
    unittest.main()
