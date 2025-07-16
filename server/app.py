# app.py
from flask import Flask, request, jsonify,  render_template
import os
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
from dataclasses import dataclass
from typing import List, Optional
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the API key
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=GOOGLE_API_KEY)  # Add this line

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
#GEMINI_API_KEY = "AIzaSyD1hnvASXHhBOvhqnRi2W_y7fmih20k2z4"
#GEMINI_API_KEY= genai.configure(api_key="AIzaSyD1hnvASXHhBOvhqnRi2W_y7fmih20k2z4")


#OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

@dataclass
class Node:
    id: str
    label: Optional[str] = None

@dataclass
class Edge:
    source: str
    target: str
    label: Optional[str] = None

@dataclass
class Subgraph:
    title: str
    nodes: List[Node]

@dataclass
class MermaidDiagram:
    nodes: List[Node]
    title: Optional[str] = None

def generate_mermaid(diagram: MermaidDiagram) -> str:
    def node_to_mermaid(node):
        if isinstance(node, Node):
            return f'{node.id}["{node.label or node.id}"]'
        elif isinstance(node, Edge):
            return f'{node.source} -->{f"|{node.label}|" if node.label else ""} {node.target}'
        elif isinstance(node, Subgraph):
            inner_content = "\n".join([node_to_mermaid(n) for n in node.nodes])
            return f"subgraph {node.title}\n{inner_content}\nend"
    
    content = "\n".join([node_to_mermaid(node) for node in diagram.nodes])
    return f"flowchart TD\n{content}"  # Remove title line

def parse_llm_response(text: str, title: str) -> MermaidDiagram:
    lines = text.strip().split('\n')
    nodes = []
    edges = []
    
    # Process only lines containing nodes or edges
    for line in lines:
        if '-->' in line:
            parts = line.split('-->')
            source = parts[0].strip()
            target = parts[1].strip()
            edges.append(Edge(source=source, target=target))
        elif '[' in line and not line.startswith('flowchart'):
            node_id = line.split('[')[0].strip()
            label = line.split('[')[1].split(']')[0].strip('"')
            nodes.append(Node(id=node_id, label=label))

    return MermaidDiagram(nodes=nodes + edges, title=title)



model = GenerativeModel('gemini-2.0-flash-exp')
sys_prompt = """

Role & Objective:
You are an instructor focused on creating Fault Tree Analysis (FTA) diagrams using mermaid.js. Your task is to visually represent failure scenarios and link them using binary logic gates (AND, OR, etc.).

Guidelines for Fault Tree Analysis (FTA) Diagrams:

Start Directly with flowchart TD

Begin with flowchart TD and avoid introductory lines or unnecessary text.
Represent Failure Scenarios

Each failure scenario should be modeled as a node.
Show logical dependencies between events (failures) using appropriate binary gates (AND, OR, etc.) .
Use Binary Logic Gates

AND gates: Use to represent failures that require multiple conditions to occur simultaneously.
OR gates: Use to show multiple potential failure causes that can trigger a single event.
Other gates (NOT, XOR, etc.) should be used as needed for specific failure logic.
Ensure Clear and Detailed Paths


The flowchart should clearly depict all failure scenarios, starting from the top event (the main failure) and showing the path to each contributing factor, with detailed connections between nodes.
Each failure scenario should be logically connected, using gates to bind the causes and effects.
Use Relevant Emojis

Use emojis to represent failure events and conditions for clearer, more intuitive understanding.
No Extra Elements

Avoid comments, unnecessary labels, or excessive descriptions.
Instead of "END", use "Exit" to signify the termination of the analysis.

flowchart TD  
A["🚨 System Failure"]  
B["🔋 Power Failure"]  
C["💻 Component Failure"]  
D["⚡ Short Circuit"]  
E["🔌 Loose Connection"]  

"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generatediagrammm():
    try:
        app.logger.debug("Received request: %s", request.json)
        data = request.get_json()
        user_input = data.get('input', '')
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if not GOOGLE_API_KEY:
            return jsonify({'error': 'API key not found'}), 500
        
        chat = model.start_chat(history=[])
        response = chat.send_message(
            f"{sys_prompt}\nCreate a diagram for: {user_input}"
        )
        
        if response.text:
            # Parse the response and generate Mermaid code
            diagram = parse_llm_response(response.text, user_input)
            mermaid_code = generate_mermaid(diagram)
            return jsonify({
                'code': mermaid_code,
                'rawResponse': response.text
            })
        
        return jsonify({'error': 'No response generated'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)