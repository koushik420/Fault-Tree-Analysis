from flask import Flask, request, jsonify, render_template
from dataclasses import dataclass
from typing import List, Optional
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API key
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Configure logging
#logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger(__name__)

# Data classes for diagram and requirements
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
class MermaidDiagram:
    nodes: List[Node]
    edges: List[Edge]
    title: Optional[str] = None

@dataclass
class SafetyRequirement:
    node_id: str
    requirement: str
    standard: str
    verification: str

# System prompt for Gemini
sys_prompt = """
You are a safety engineering expert specializing in Fault Tree Analysis (FTA) and functional safety. 
When analyzing systems, follow these steps:

1. Generate a detailed Fault Tree Analysis diagram using Mermaid.js syntax with proper hierarchy and logical gates.
2. For each node in the FTA, identify whether the failure is hardware-related, software-related, or a combination of both, generate specific safety requirements including:
   - Prevention/mitigation measures
   - Relevant safety standards (IEC 61508 (functional safety), ISO 26262 (ASIL for automotive software), ISO 21434 etc.)
   - Verification methods (testing, analysis, inspection)
   - Safety integrity levels (SIL/ASIL) where applicable

Format requirements using this structure:
### Requirements
1. [Node ID]: [Clear requirement text]
   - Standard: [Applicable standard]
   - Verification: [Verification method]
   - Safety Level: [SIL/ASIL level if applicable]

### Guidelines for Fault Tree Analysis (FTA) Diagrams:
Role & Objective:
You are an instructor focused on creating Fault Tree Analysis (FTA) diagrams using mermaid.js. Your task is to visually represent failure scenarios and link them using binary logic gates (AND, OR, etc.).

Start Directly with flowchart TD

Begin with flowchart TD and avoid introductory lines or unnecessary text.
Represent Failure Scenarios

Each failure scenario should be modeled as a node.
Show logical dependencies between events (failures) using appropriate binary gates (AND, OR, etc.) .
Use Binary Logic Gates

AND gates: Use to represent failures that require multiple conditions to osccur simultaneously.
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
Avoid showing while If you do not get the node name or not able to setup connection.

flowchart TD
    A["🚨 Brake-by-Wire System Failure"]
    B["🔧 Hardware Fault: Brake Actuator Failure"]
    C["💻 Software Error: Control Algorithm Fault"]
    D["⚡ Electrical Fault: Power Supply Failure"]
    E["🔌 Communication Error: CAN Bus Failure"]
    F["🔥 Overheating: Brake Actuator Overload"]
    G["🔧 Mechanical Fault: Brake Pad Wear"]
    H["💻 Software Bug: Sensor Data Processing Error"]
    I["⚡ Environmental Factor: Water Ingress"]
    J["🔧 Hardware Fault: Sensor Failure"]
    K["💻 Software Error: Memory Leak in Control Software"]
    L["⚡ Electrical Fault: Wiring Harness Damage"]
    M["🔧 Mechanical Fault: Actuator Motor Failure"]
    N["💻 Software Error: Timing Issue in Real-Time OS"]
    O["⚡ Environmental Factor: Extreme Temperature"]
    P["🔧 Hardware Fault: ECU Failure"]
    Q["💻 Software Error: Stack Overflow"]
    R["⚡ Electrical Fault: Ground Fault"]
    S["🔧 Mechanical Fault: Actuator Gear Wear"]
    T["💻 Software Error: Deadlock in Multithreading"]
    U["⚡ Environmental Factor: Vibration"]
    
    A -- OR --> B
    A -- OR --> C
    A -- OR --> D
    B -- OR --> F
    B -- OR --> G
    B -- OR --> J
    B -- OR --> M
    B -- OR --> P
    C -- OR --> E
    C -- OR --> H
    C -- OR --> K
    C -- OR --> N
    C -- OR --> Q
    C -- OR --> T
    D -- OR --> I
    D -- OR --> L
    D -- OR --> O
    D -- OR --> R
    D -- OR --> U
    F -- AND --> V["⚡ High Current Draw"]
    F -- AND --> W["🔧 Poor Heat Dissipation"]
    G -- AND --> X["🔧 Low-Quality Brake Pad Material"]
    G -- AND --> Y["⚡ Excessive Braking Force"]
    J -- AND --> Z["🔧 Sensor Calibration Error"]
    J -- AND --> AA["⚡ Sensor Signal Noise"]
    M -- AND --> AB["🔧 Motor Bearing Wear"]
    M -- AND --> AC["⚡ Motor Driver Fault"]
    P -- AND --> AD["🔧 ECU Cooling Failure"]
    P -- AND --> AE["⚡ Voltage Spike"]
    Q -- AND --> AF["💻 Insufficient Stack Size"]
    Q -- AND --> AG["💻 Recursive Function Error"]
    T -- AND --> AH["💻 Improper Thread Synchronization"]
    T -- AND --> AI["💻 Resource Contention"]

### Requirements
1. B: Implement redundant brake actuators with real-time monitoring
   - Standard: ISO 26262 (ASIL D)
   - Verification: Hardware-in-loop testing and fault injection
   - Safety Level: ASIL D

2. C: Use formal methods to verify the control algorithm
   - Standard: ISO 26262 (ASIL D)
   - Verification: Model-based testing and code review
   - Safety Level: ASIL D

3. D: Install redundant power supply with surge protection
   - Standard: ISO 26262 (ASIL C)
   - Verification: Electrical stress testing
   - Safety Level: ASIL C

4. E: Implement error detection and correction in CAN communication
   - Standard: ISO 26262 (ASIL C)
   - Verification: CAN protocol testing and fault injection
   - Safety Level: ASIL C

5. F: Add thermal sensors and automatic load reduction
   - Standard: ISO 26262 (ASIL B)
   - Verification: Thermal cycling and overload testing
   - Safety Level: ASIL B

6. G: Use wear-resistant materials and real-time wear monitoring
   - Standard: ISO 26262 (ASIL B)
   - Verification: Durability testing and field data analysis
   - Safety Level: ASIL B

7. H: Implement data validation and redundancy in sensor processing
   - Standard: ISO 26262 (ASIL C)
   - Verification: Fault injection and simulation
   - Safety Level: ASIL C

8. I: Use waterproof enclosures and connectors
   - Standard: ISO 26262 (ASIL B)
   - Verification: Environmental testing (IP67)
   - Safety Level: ASIL B

9. J: Use high-reliability sensors with built-in diagnostics
   - Standard: ISO 26262 (ASIL B)
   - Verification: Diagnostic coverage analysis
   - Safety Level: ASIL B

10. K: Implement memory management best practices and static analysis
    - Standard: ISO 26262 (ASIL C)
    - Verification: Static code analysis and memory testing
    - Safety Level: ASIL C

11. L: Use shielded wiring harnesses with strain relief
    - Standard: ISO 26262 (ASIL B)
    - Verification: Vibration and stress testing
    - Safety Level: ASIL B

12. M: Implement motor health monitoring and redundancy
    - Standard: ISO 26262 (ASIL C)
    - Verification: Motor performance testing
    - Safety Level: ASIL C

13. N: Use a certified real-time OS with timing guarantees
    - Standard: ISO 26262 (ASIL D)
    - Verification: Timing analysis and real-time testing
    - Safety Level: ASIL D

14. O: Use temperature-resistant components and thermal management
    - Standard: ISO 26262 (ASIL B)
    - Verification: Environmental stress testing
    - Safety Level: ASIL B

15. P: Implement ECU redundancy and cooling mechanisms
    - Standard: ISO 26262 (ASIL C)
    - Verification: ECU stress testing
    - Safety Level: ASIL C

16. Q: Use stack monitoring and recursion limits
    - Standard: ISO 26262 (ASIL C)
    - Verification: Stack usage analysis
    - Safety Level: ASIL C

17. T: Implement proper thread synchronization and resource management
    - Standard: ISO 26262 (ASIL D)
    - Verification: Concurrency testing
    - Safety Level: ASIL D

"""

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response into diagram and requirements."""
    try:
        # Split into diagram and requirements sections
        sections = response_text.split("### Requirements")
        mermaid_section = sections[0].strip()
        requirements_section = sections[1].strip() if len(sections) > 1 else ""

        # Parse Mermaid diagram
        nodes = []
        edges = []
        for line in mermaid_section.split('\n'):
            line = line.strip()
            if '-->' in line:
                # Extract source and target nodes
                parts = line.split('-->')
                source = parts[0].strip().split('[')[0].strip()  # Extract node ID
                target = parts[1].strip().split('[')[0].strip()  # Extract node ID
                edges.append(Edge(source=source, target=target))
            elif '[' in line and not line.startswith('flowchart'):
                # Extract node ID and label
                node_id = line.split('[')[0].strip()
                label = line.split('[')[1].split(']')[0].strip('"')  # Extract label
                nodes.append(Node(id=node_id, label=label))

        # Parse safety requirements
        requirements = []
        if requirements_section:
            lines = requirements_section.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith(tuple(str(i) for i in range(1, 10))):  # Check if line starts with a number
                    parts = line.split(':')
                    if len(parts) >= 2:
                        node_id = parts[0].split('.')[1].strip()
                        requirement = parts[1].strip()
                        standard = ""
                        verification = ""

                        # Look ahead for standard and verification
                        for j in range(i + 1, min(i + 4, len(lines))):  # Check next 3 lines
                            next_line = lines[j].strip()
                            if next_line.startswith("- Standard:"):
                                standard = next_line.split(":")[1].strip()
                            elif next_line.startswith("- Verification:"):
                                verification = next_line.split(":")[1].strip()

                        requirements.append({
                            "node_id": node_id,
                            "requirement": requirement,
                            "standard": standard,
                            "verification": verification
                        })
                        i += 1  # Move to the next line
                    else:
                        i += 1
                else:
                    i += 1

        return {
            "diagram": MermaidDiagram(nodes=nodes, edges=edges),
            "requirements": requirements
        }
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        raise

def generate_mermaid(diagram: MermaidDiagram) -> str:
    """Generate Mermaid.js code from diagram structure."""
    nodes_str = "\n".join(
        f'{node.id}["{node.label}"]'  # Use descriptive labels
        for node in diagram.nodes
    )
    edges_str = "\n".join(
        f'{edge.source} --> {edge.target}' 
        for edge in diagram.edges
    )
    return f"flowchart TD\n{nodes_str}\n{edges_str}"

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_diagram():
    try:
        data = request.get_json()
        user_input = data.get('input', '')
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        # Generate response using Gemini
        chat = model.start_chat(history=[])
        response = chat.send_message(
            f"{sys_prompt}\nCreate a diagram for: {user_input}"
        )

        if not response.text:
            return jsonify({'error': 'No response generated by the model'}), 500

        # Parse the response and generate Mermaid code
        parsed_response = parse_llm_response(response.text)
        mermaid_code = generate_mermaid(parsed_response["diagram"])

        return jsonify({
            'mermaid_code': mermaid_code,
            'requirements': parsed_response["requirements"],
            'raw_response': response.text
        })

    except Exception as e:
        logger.error(f"Error in generate_diagram: {e}")
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)