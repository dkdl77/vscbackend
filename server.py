from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)

@app.route('/search')
def search():
    # 여기에 객체 탐지/이미지 반환 코드
    # 예시: return send_file('detected_people/person_1.jpg', mimetype='image/jpeg')
    pass

if __name__ == '__main__':
    app.run()