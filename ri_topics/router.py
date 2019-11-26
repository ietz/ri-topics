from flask import Flask, escape, request, jsonify

from ri_topics.mock_data import generate_top_trend, generate_flop_trend

app = Flask(__name__)


@app.route('/<account_name>')
def hello(account_name: str):
    return jsonify({
        'top': [generate_top_trend() for _ in range(3)],
        'flop': [generate_flop_trend() for _ in range(3)],
    })

