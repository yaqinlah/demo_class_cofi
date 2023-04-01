from flask import Flask, render_template, request
from utils import predict_text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_text(text)
        return render_template('main.html', text=text, prediction=prediction)
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
