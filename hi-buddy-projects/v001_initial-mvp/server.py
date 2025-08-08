
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "MVP server running locally"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
