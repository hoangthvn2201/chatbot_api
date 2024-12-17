import os
from flask import Flask, render_template, request, url_for, redirect, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from flask import Blueprint
from flask_cors import CORS
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from pyngrok import ngrok
import pyodbc
from functools import wraps
import datetime
from typing import List, Dict, Union, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re
import pandas as pd

app = Flask(__name__, static_folder='static')
CORS(app)

app.config['SECRET_KEY'] = 'secret-key-goes-here'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

MODEL_PATH = "huyhoangt2201/1b_multitableJidouka_new_merged"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main = Blueprint('main', __name__)

# Uncomment and update with your actual database connection details
def get_db_connection():
    conn_str = (
        r'DRIVER={SQL Server};'
        r'SERVER=10.73.131.12;'
        r'DATABASE=JidoukaProject;'
        r'UID=intern;'
        r'PWD=intern1234qwer!'
    )
    return pyodbc.connect(conn_str)

class ExecuteQuery:
    def __init__(self, db):
        self.db = db
    
    def is_valid_sql_query(self, query: str) -> bool:
        # Basic validation to ensure it's a SELECT query
        return query.upper().startswith('SELECT')
    
    def sanitize_query(self, query: str) -> str:
        # Remove comments and unnecessary whitespace
        query = re.sub(r'/\*.*?\*/', '', query)
        query = re.sub(r'--.*$','', query)
        query = query.strip()
        
        # Replace column names if needed
        query = query.replace('ImprovementName', 'ImprovementContent')
        query = query.replace('TimeSaving', 'TotalTimeSaved')
        
        return query
    
    def execute_query(self, query: str) -> Union[List[Dict[str, Any]], str]:
        try:
            if self.is_valid_sql_query(query):
                sanitized_query = self.sanitize_query(query)
                # Use pandas to read SQL and convert to dictionary
                df = pd.read_sql_query(sanitized_query, self.db)
                if len(df.columns) == 1:
                    return f"{df.columns[0]}:{df.iloc[0]}"
                # Convert to list of dictionaries for JSON serialization
                return df.to_dict('records')
            else:
                return "Invalid SQL query"
        except Exception as e:
            return f"Error executing query: {str(e)}"

class JidoukaModel:
    def __init__(self, max_history: int=0):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
    
    def _build_prompt(self) -> str:
        # Existing prompt building logic remains the same
        system_prompt = """You are an SQL query assistant. Based on schema, generate an SQL query to retrieve the relevant information for the user. If the user's question is unrelated to the table, respond naturally in user's language.
        Schema:
        +Table Author, columns=[AuthorId: int, AuthorName: nvarchar(255), DepartmentId int, GroupDCId int]
        +Table Department, columns=[DepartmentId: int, DepartmentName: nvarchar(255)]
        +Table GroupDC, columns=[GroupDCId: int, DepartmentId: int, GroupDCName nvarchar(255)]
        +Table Job, columns=[JobId: int, JobName: nvarchar(255)]
        +Table Tool, columns=[ToolId: int, ToolName: nvarchar(255), ToolDescription: text]
        +Table Jidouka, columns=[JidoukaId: bigint, ProductApply: nvarchar(255), ImprovementName: nvarchar(255), SoftwareUsing: nvarchar(255), Description: nvarchar(255), Video: text, DetailDocument: text, TotalJobApplied: int, TotalTimeSaved: int, DateCreate: datetime, JobId: int, AuthorId: int, DepartmentId: int, GroupDCId: int]
        +Table JidoukaTool, columns=[JidoukaId: bigint, ToolId: int]
        +Primary_keys=[Author.AuthorId, Department.DepartmentId, GroupDC.GroupDCId, Job.JobId, Tool.ToolId, Jidouka.JidoukaId]
        +Foreign_keys=[GroupDC.DepartmentId=Department.DepartmentId, Jidouka.JobId=Job.JobId, Jidouka.AuthorId=Author.AuthorId, Jidouka.DepartmentId=Department.DepartmentId, Jidouka.GroupDCId=GroupDC.GroupDCId, JidoukaTool.JidoukaId=Jidouka.JidoukaId, JidoukaTool.ToolId=Tool.ToolId, Author.DepartmentId=Department.DepartmentId, Author.GroupDCId=GroupDC.GroupDCId]
        """
        return system_prompt
    
    def chat(self, user_input: str) -> str:
        # Existing chat method remains the same
        prompt = self._build_prompt()
        eot = "<|eot_id|>"
        eot_id = self.tokenizer.convert_tokens_to_ids(eot)
        self.tokenizer.pad_token = eot
        self.tokenizer.pad_token_id = eot_id

        messages =[
            {'role':'system', 'content':prompt},
            {'role':'user', 'content':user_input}
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = self.model.generate(inputs['input_ids'],
                                      attention_mask=inputs['attention_mask'],
                                      temperature = 0.1, 
                                      do_sample = True,
                                      top_p = 0.1,
                                      max_new_tokens=512).to(DEVICE)
        bot_response = self.tokenizer.decode(outputs[0])
        bot_response = bot_response.split('<|start_header_id|>assistant<|end_header_id|>')
        bot_response = bot_response[1].strip()[:-10]
        
        self.conversation_history.append({
            'human': user_input,
            'assistant': bot_response
        })
        
        return bot_response

# Initialize objects
chatbot = JidoukaModel()
mydb = get_db_connection()
query_agent = ExecuteQuery(mydb)
chat_history = []

@main.route('/')
def chat_page():
    return render_template('chat_page.html', history=[])

@main.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Get SQL query from chatbot
    sql_query = chatbot.chat(user_message)
    
    # Execute SQL query
    query_result = query_agent.execute_query(sql_query)
    
    # Prepare response
    if isinstance(query_result, list):
        # If query returns results, convert to readable string
        response = "\n".join([str(record) for record in query_result])
    else:
        # If query fails or returns error
        response = query_result

    chat_history.append({'timestamp': timestamp, 'bot': response})

    return jsonify({
        'timestamp': timestamp, 
        'response': response,
        'query': sql_query  # Optional: send back the generated SQL query
    })

@main.route('/clear', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.register_blueprint(main)
    app.run(host='10.73.131.32', port=7860)
