#!/usr/bin/env python3
"""
Simple development server for OpenStatica frontend
"""

import http.server
import socketserver
import os
import sys

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def run_server():
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🚀 OpenStatica Frontend Server")
        print(f"📁 Serving: {DIRECTORY}")
        print(f"🌐 URL: http://localhost:{PORT}")
        print(f"📊 Test: http://localhost:{PORT}/test.html")
        print(f"\nPress Ctrl+C to stop...")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            sys.exit(0)


if __name__ == "__main__":
    os.chdir(DIRECTORY)
    run_server()
