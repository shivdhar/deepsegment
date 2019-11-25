import flask
from flask import Flask, jsonify, request
from deepsegment import DeepSegment

app = Flask(__name__)

segmenter = DeepSegment('en')

@app.route('/predict', methods=['POST'])
def get_preds():
    sents = request.json
    assert isinstance(sents, (str, list))
    return jsonify(segmenter.segment(sents))

if __name__ == '__main__':
    app.run('0.0.0.0', 12000, threaded=False)
