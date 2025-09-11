#!/usr/bin/env python3
"""
Start the WET Pipeline Web UI

This script starts the Flask web application for semantic search.
"""

import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import psycopg
        import sentence_transformers
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_database():
    """Check if database is accessible."""
    try:
        sys.path.append(str(Path(__file__).parent / "scripts"))
        from db import get_conn, init_db
        
        init_db()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please ensure PostgreSQL is running and accessible")
        return False

def main():
    """Main function to start the web UI."""
    print("🚀 Starting WET Pipeline Web UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check database
    if not check_database():
        sys.exit(1)
    
    print("=" * 50)
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n👋 Web UI stopped")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
