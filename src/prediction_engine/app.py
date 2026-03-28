from flask import Flask, render_template
from predict import predict_bp   
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


app.register_blueprint(predict_bp,url_prefix='/predict') 


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
