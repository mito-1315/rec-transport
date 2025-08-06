#!/usr/bin/env python3
"""
Simple runner script for CSV Coordinate Viewer
"""

import os
import sys
import socket

def check_port_available(port):
    """Check if a port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        return True
    except OSError:
        sock.close()
        return False

def find_available_port(start_port=5000):
    """Find an available port starting from start_port"""
    ports_to_try = [5000, 5001, 8000, 8080, 3000, 3001, 9000, 8888]
    
    for port in ports_to_try:
        if check_port_available(port):
            return port
    
    # If none of the preferred ports work, try a range
    for port in range(5002, 5050):
        if check_port_available(port):
            return port
    
    return None

def main():
    """Run the CSV coordinate viewer"""
    print("📍 CSV Coordinate Point Viewer")
    print("=" * 40)
    
    # Check if required packages are installed
    try:
        import flask
        import pandas
        import folium
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Find an available port
    port = find_available_port()
    if port is None:
        print("❌ Could not find an available port")
        print("Please close some applications and try again")
        sys.exit(1)
    
    # Import and run the main application
    try:
        from csv_coordinate_viewer import app
        
        if port != 5000:
            print(f"⚠️  Port 5000 is busy, using port {port} instead")
        
        print(f"\n🚀 Starting coordinate point viewer on port {port}...")
        print(f"📍 Open your browser and go to:")
        print(f"   • http://localhost:{port}")
        print(f"   • http://127.0.0.1:{port}")
        print("⏹️  Press Ctrl+C to stop the server\n")
        
        app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure no other Flask apps are running")
        print("2. On macOS, disable AirPlay Receiver in System Preferences")
        print("3. Try running: lsof -ti:5000 | xargs kill -9")
        sys.exit(1)

if __name__ == "__main__":
    main()
