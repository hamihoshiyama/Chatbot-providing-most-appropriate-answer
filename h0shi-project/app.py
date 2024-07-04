from flask import Flask, render_template, redirect, url_for, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
import openai
import torch
import transformers
from transformers import BertJapaneseTokenizer, BertModel, BertForSequenceClassification, AdamW
import numpy as np
import pandas as pd
import json
import requests
import random
import time
import os
import json
import re 
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import ipadic 
from cryptography.fernet import Fernet

from dotenv import load_dotenv
#環境変数.envからデータを読み込む
load_dotenv()
import secrets
#32バイトの乱数を16進数で表現した文字列（64文字）を生成==>暗号キー
secret_key = secrets.token_hex(32)

#Flaskアプリケーションのインスタンスを作成
app = Flask(__name__)  # _name_は現在のモジュール名
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# 暗号化キーの生成または読み込み
if not os.path.exists('secret.key'):
    key = Fernet.generate_key()
    with open('secret.key', 'wb') as key_file:
        key_file.write(key)
else:
    with open('secret.key', 'rb') as key_file:
        key = key_file.read()

cipher = Fernet(key)

openai.api_key= os.getenv("OPENAI_API_KEY")
slack_token_bot =None
slack_token_user = None

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
classification_model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
classification_model.to(device)

fine_tuned_models={}
status="idle"#ステータスが「待機中」
classifier_model=None
validation_file_paths=[]
model_user_names=[]


class APIKey(db.Model):  # APIキーを保存するテーブル定義
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    bot_token = db.Column(db.String(200), nullable=False)
    user_token = db.Column(db.String(200), nullable=False)

@app.route('/submit_keys', methods=['POST'])
def submit_keys():
    user_id = request.form['user_id']
    bot_token = request.form['bot_token']
    user_token = request.form['user_token']

    # APIキーの暗号化
    encrypted_bot_token = cipher.encrypt(bot_token.encode())
    encrypted_user_token = cipher.encrypt(user_token.encode())

    # データベースに保存
    new_key = APIKey(user_id=user_id, bot_token=encrypted_bot_token, user_token=encrypted_user_token)
    db.session.add(new_key)
    db.session.commit()

    return 'API keys submitted successfully'

@app.route('/get_keys', methods=['GET'])
def get_keys():
    user_id = request.args.get('user_id')
    api_key = APIKey.query.filter_by(user_id=user_id).first()
    if api_key:
        # APIキーの復号化
        bot_token = cipher.decrypt(api_key.bot_token).decode()
        user_token = cipher.decrypt(api_key.user_token).decode()
        # グローバル変数に設定
        global slack_token_bot, slack_token_user
        slack_token_bot = bot_token
        slack_token_user = user_token
        return jsonify({"bot_token": bot_token, "user_token": user_token})
    else:
        return 'API keys not found'

def initialize_slack_client():
    global client, headers
    client = WebClient(token=slack_token_user)
    headers = {'Authorization': f'Bearer {slack_token_bot}'}

@app.route('/use_keys', methods=['GET'])
def use_keys():
    # APIキーを使用するためにSlackクライアントを初期化
    initialize_slack_client()
    return 'Slack client initialized with new tokens'
   
def get_user_id(user_name, headers):
    try:
        while True:
            response = requests.get('https://slack.com/api/users.list', headers=headers)
            if response.status_code == 429:  # レート制限
                retry_after = int(response.headers.get('Retry-After', 1))
                print(f"Rate limited. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                response.raise_for_status()
                users = response.json().get('members', [])
                for user in users:
                    profile = user.get('profile', {})
                    if profile.get('display_name') == user_name:
                        return user['id']
                break

    except requests.exceptions.RequestException as e:
        print(f"Error fetching user list: {e}")
    return None

def get_channel_id(channel_name):
    try:
        while True:
            response = requests.get('https://slack.com/api/conversations.list', headers=headers)
            if response.status_code == 429:  # レート制限
                retry_after = int(response.headers.get('Retry-After', 1))
                print(f"Rate limited. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                response.raise_for_status()
                channels = response.json().get('channels', [])
                for channel in channels:
                    if channel['name'].strip() == channel_name.strip():
                        return channel['id']
                break
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channel list: {e}")
    return None

def get_dm_channel_id(user_id):
    try:
        response = client.conversations_open(users=[user_id])
        if response['ok']:
            return response['channel']['id']
        else:
            print("Failed to open or fetch DM channel:", response['error'])
            return None
    except SlackApiError as e:
        print(f"An error occurred: {e.response['error']}")
        return None
    

def fetch_all_messages(channel_id, num_messages):
    messages = []
    has_more = True
    latest = datetime.now().timestamp()
    while has_more and len(messages) < num_messages:
        try:
            response = client.conversations_history(
                channel=channel_id,
                latest=str(latest),
                limit=min(1000, num_messages - len(messages))  # 一度に取得するメッセージ数を最大に設定
            )
            if response['ok']:
                messages.extend(response['messages'])
                has_more = response['has_more']
                if has_more:
                    latest = response['messages'][-1]['ts']
            else:
                has_more = False
        except SlackApiError as e:
            print(f"Slack API error occurred: {e.response['error']}")
            has_more = False
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            has_more = False
    return messages[:num_messages]

def format_messages_to_jsonl(messages, model_user_name, model_id, my_id, filename):
    with open(filename, "w", encoding='utf-8') as f:
        system_message = {"role": "system", "content": f"あなたは{model_user_name}です"}
        user_message = None
        assistant_message = None
        for message in messages:
            text = message.get("text")
            content = re.sub(r'<@[^>]+>', '', text)
            user = message.get("user")
            if user == model_id:
                assistant_message = {"role": "assistant", "content": content}
            elif user == my_id:
                user_message = {"role": "user", "content": content}
            if user_message and assistant_message:
                combined_messages = [system_message, user_message, assistant_message]
                f.write(json.dumps({"messages": combined_messages}, ensure_ascii=False) + "\n")
                user_message = None
                assistant_message = None

def load_and_shuffle_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file.readlines()]
    random.shuffle(data)
    return data

def split_data(data, train_ratio=0.9):
    if len(data) > 300:
        data = data[:300]
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    while len(train_data) < 100:
        train_data.extend(train_data[:100 - len(train_data)])
    return train_data, test_data

def save_data_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def fine_tune_model(train_data, test_data, api_token):
    openai.api_key = api_token

    train_file= openai.files.create(
        file=open(train_data, "rb"),
        purpose="fine-tune"
    )
    validation_file = openai.files.create(
        file=open(test_data, "rb"),
        purpose="fine-tune"
    )

    response = openai.fine_tuning.jobs.create(
        model="gpt-3.5-turbo",
        training_file=train_file.id,
        validation_file=validation_file.id,
        hyperparameters={
          "n_epochs":3
        }
    )
    return response

def truncate_text(text):
    return text[:512]

def sentence_to_vector(model, tokenizer, sentence):
    tokens = tokenizer(sentence)["input_ids"]
    input = torch.tensor(tokens).reshape(1, -1)
    with torch.no_grad():
        outputs = model(input, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state[0]
        averaged_hidden_state = last_hidden_state.sum(dim=0) / len(last_hidden_state)
    return averaged_hidden_state

def calc_similarity(sentence1, sentence2):
    sentence_vector1 = sentence_to_vector(bert_model, tokenizer, sentence1)
    sentence_vector2 = sentence_to_vector(bert_model, tokenizer, sentence2)
    score = torch.nn.functional.cosine_similarity(sentence_vector1, sentence_vector2, dim=0).detach().numpy().copy()
    return score

def chat_with_finetuned_model(prompt, model_id):
    response = openai.chat.completions.create(
        model=model_id,
        messages=[
          {"role": "system", "content": "あなたは{model}です"},
          {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].text.strip()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prepare_models', methods=['POST'])
def prepare_models():
    global status, validation_file_paths, model_user_names
    data = request.get_json()
    my_name=data["my_user_name"]
    model_users = data['model_users']
    channel = data['channel']

    status = "preparing"
    for model in model_users:
        model_id = get_user_id(model, headers)
        if not model_id:
            return jsonify({"error": f"User {model} not found"}), 404
        
        channel_id = get_channel_id(channel)
        if not channel_id:
            return jsonify({"error": f"Channel {channel} not found"}), 404

        dm_channel_id = get_dm_channel_id(model_id)
        if not dm_channel_id:
            return jsonify({"error": f"DM channel for user {model} not found"}), 404
        
        messages = fetch_all_messages(dm_channel_id, 550)
        if messages:
            filename = f"./data/training_data_{model}.jsonl"
            data = load_and_shuffle_data(filename)
            train_data, test_data = split_data(data)
            train_file_path = f"./data/{model}_train.jsonl"
            validation_file_path = f"./data/{model}_validation.jsonl"
            save_data_to_file(train_data, train_file_path)
            save_data_to_file(test_data, validation_file_path)
            fine_tune_model(train_file_path, validation_file_path)
            fine_tuned_models[model] = filename
            validation_file_paths.append(validation_file_path)
            model_user_names.append(model)
    
        else:
            return jsonify({"error": f"No messages found for user {model}"}), 404
        
    status = "tuning"
    return jsonify({"message": "Models prepared"})


@app.route('/check_status', methods=['GET'])
def check_status():
    global status
    if status == "tuning":
        # Simulate fine-tuning completion
        time.sleep(10)  # Replace with actual fine-tuning check
        status = "completed"
    return jsonify({"status": status})

@app.route('/ask_models', methods=['POST'])
def ask_models():
    data = request.get_json()
    question = data['question']
    responses = []

    for model, model_id in fine_tuned_models.items():
        response = openai.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "あなたは{model}です"},
                {"role": "user", "content": question}
            ],
            max_tokens=150
        )
        responses.append(response.choices[0].text.strip())
    results=[]
    for i, (validation_file_path, model_name) in enumerate(zip(validation_file_paths, list(fine_tuned_models.values()))):
        with open(validation_file_path, 'r', encoding='utf-8') as f:
            validation_data = [json.loads(line) for line in f]

        for item in validation_data:
            messages = item['messages']
            prompt = next(message['content'] for message in messages if message['role'] == 'user')
            actual_response = next(message['content'] for message in messages if message['role'] == 'assistant')
            prompt = truncate_text(prompt)
            actual_response = truncate_text(actual_response)
            for j, model_name in enumerate(list(fine_tuned_models.values())):
                generated_response = chat_with_finetuned_model(prompt, model_name)
                similarity = calc_similarity(actual_response, generated_response)

                results.append({
                    "Validation_File": f"{model_user_names[i]}_validation_data",
                    "Model": f"{model_user_names[j]}",
                    "Prompt": prompt,
                    "Actual Response": actual_response,
                    "Generated Response": generated_response,
                    "Cosine Similarity": similarity
                })

    # Simulate cosine similarity calculation
    df = pd.DataFrame(results)
    df = df[["Validation_File", "Model", "Cosine Similarity"]]

    df_avg = df.groupby(['Validation_File', 'Model']).mean().reset_index()
    df_avg.rename(columns={'Cosine Similarity': 'Cosine Similarity Average'}, inplace=True)

    return jsonify({"responses": responses, "similarity": df_avg.to_dict(orient="records")})

@app.route('/create_classifier', methods=['POST'])
def create_classifier():
    global status, classifier_model
    status = "creating_classifier"
    
    # Simulate classifier creation
    # Load and prepare data for classifier training
    data = []
    for user in fine_tuned_models.keys():
        file_path = f"./data/{user}_train.jsonl"
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
    
    # Split data into training and validation
    train_data, val_data = split_data(data, train_ratio=0.9)
    
    # Train classifier (this is a simulation, replace with actual training)
    time.sleep(5)  # Replace with actual classifier creation
    
    classifier_model = "simulated_classifier_model"  # Placeholder for actual classifier model
    
    status = "classifier_created"
    return jsonify({"message": "Classifier created"})

@app.route('/classify_and_respond', methods=['POST'])
def classify_and_respond():
    data = request.get_json()
    text = data['text']

    # 分類器を使用してモデルを選択
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = classifier_model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    selected_model_user = model_user_names[predicted_label]

    # 選択されたモデルを使用して応答を生成
    model_id = fine_tuned_models[selected_model_user]
    response_text = chat_with_finetuned_model(text, model_id)

    # 精度評価
    similarity_scores = []
    for i, validation_file_path in enumerate(validation_file_paths):
        with open(validation_file_path, 'r', encoding='utf-8') as f:
            validation_data = [json.loads(line) for line in f]

        for item in validation_data:
            messages = item['messages']
            prompt = next(message['content'] for message in messages if message['role'] == 'user')
            actual_response = next(message['content'] for message in messages if message['role'] == 'assistant')
            prompt = truncate_text(prompt)
            actual_response = truncate_text(actual_response)
            generated_response = chat_with_finetuned_model(prompt, model_id)
            similarity = calc_similarity(actual_response, generated_response)
            similarity_scores.append(similarity)

    avg_similarity = np.mean(similarity_scores)

    return jsonify({"response": response_text, "similarity": avg_similarity})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="127.0.0.1", port=8080)

