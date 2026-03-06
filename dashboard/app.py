import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
from flask import Flask, send_from_directory, render_template

app = Flask(__name__)

# Base directory for the dashboard
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_dashboard')

@app.route('/')
def index():
    """Serve the main index.html file."""
    return send_from_directory(DASHBOARD_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (JSON, JS, CSS) from the dashboard directory."""
    return send_from_directory(DASHBOARD_DIR, filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve images and other assets from the assets subdirectory."""
    assets_dir = os.path.join(DASHBOARD_DIR, 'assets')
    return send_from_directory(assets_dir, filename)

if __name__ == '__main__':
    print("="*50)
    print("DEBRECEN WEB GIS SERVER")
    print("="*50)
    print(f"Server Directory: {DASHBOARD_DIR}")
    print("Access the dashboard at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server.")
    print("="*50)
    

