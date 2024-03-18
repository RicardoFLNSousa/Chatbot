from flask import Flask, request, render_template
import simple_model as sm
import seq_2_seq_model as s2s
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/process_question', methods=['POST'])
def process_question():
    user_question = request.form['question']

    response_s2s = s2s.generate_response(user_question, os.getcwd()+"/models/")
    response_sm = sm.generate_response(user_question, os.getcwd()+"/models/")

    processed_response_sm = f'Pergunta: {user_question}. Resposta do Simple Model: {response_sm}'
    processed_response_s2s = f'Pergunta: {user_question}. Resposta do Seq2Seq Model: {response_s2s}'
    return render_template('form.html', response=processed_response_sm, response2=processed_response_s2s)

if __name__ == '__main__':
    app.run(debug=True)
