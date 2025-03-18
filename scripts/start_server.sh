#!/bin/bash

# Application directory
APP_DIR="/home/ubuntu/myproject"
LOG_FILE="$APP_DIR/app_output.log"

# Check for existing Flask application processes
echo "Checking for existing application processes..."
EXISTING_PID=$(pgrep -f "python3 app.py")

if [ -n "$EXISTING_PID" ]; then
    echo "Found running application process with PID: $EXISTING_PID. Terminating it..."
    kill -9 "$EXISTING_PID"
    echo "Terminated previous application process."
else
    echo "No existing application processes found."
fi

# Navigate to the application directory
cd "$APP_DIR"

# Set up virtual environment
echo "Setting up the virtual environment..."

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

# Start the  application with nohup
echo "Starting application..."
nohup python3 app.py > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "Application started with PID: $NEW_PID"