import subprocess
from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def train_model_https(request):
    try:
        # Log the received request
        print(f"Received request: {request.json}")

        # Run the training script
        result = subprocess.run(['python', 'training.py'], capture_output=True, text=True)
        
        # Log the output
        print(result.stdout)
        
        # Check if there was an error
        if result.returncode != 0:
            print(result.stderr)
            return f"Error: {result.stderr}", 500
        
        return "Training started successfully", 200
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
