# LangGraph Email Support Agent — Tutorial Implementation

# This script demonstrates how to build a multi-step AI agent using LangGraph.
# The agent simulates an automated email support workflow with classification, parallel processing, and human-in-the-loop approval.

# Core Flow:
#   1. Ingest raw email input
#   2. Classify the email (intent, urgency, topic) using an LLM
#   3. In parallel:
#        - Search internal documentation for relevant answers
#        - Create a bug ticket (simulated)
#   4. Generate a draft response using all available context
#   5. Conditionally:
#        - Send automatically (low risk)
#        - OR pause for human approval (high urgency / complex cases)

# Key LangGraph Concepts Demonstrated:
#   - State management using TypedDict
#   - Structured LLM outputs mapped to state
#   - Parallel node execution (fan-out / fan-in pattern)
#   - Conditional routing using Command(goto=...)
#   - Human-in-the-loop using interrupt() and resume
#   - Persistence via thread_id and checkpointer
#   - Batch processing with queued approvals

# Notes:
#   - External integrations (search, bug tracking, email sending) are mocked
#   - Designed for learning purposes, but can be extend for production use

import uuid, os
from typing import Literal, TypedDict

from dotenv import load_dotenv
load_dotenv() # load env vars before any library that might need them, ie ChatOpenAI

from IPython.display import Image, display

from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# Define state schemas
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # raw email data
    email_content: str
    sender_email: str
    email_id: str

    # classification result
    classification: EmailClassification | None

    # bug tracking
    ticket_id: str | None

    # raw search results
    search_results: list[str] | None
    customer_history: dict | None

    # generated content
    draft_response: str | None

# Define nodes & edges
def read_email(state: EmailAgentState) -> EmailAgentState:
    """Extract and parse email content"""
    pass

llm = ChatOpenAI(model="gpt-5-mini")

def classify_intent(state: EmailAgentState) -> EmailAgentState:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    classification_prompt = f"""
    Analyse this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification, including intent, urgency, topic, and summary.
    """

    # get structured response directly as a dict
    classification = structured_llm.invoke(classification_prompt)

    # store classification as a single dict in state
    return {"classification": classification}

def search_documentation(state: EmailAgentState) -> EmailAgentState:
    """Search knowledge base for relevant information"""
 
    # build search query from classification
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        # implement search logic here
        search_results = [
                "—Search_result_1—",
                "—Search_result_2—",
                "—Search_result_3—",
        ]
    except SearchAPIError as e:
        # for recoverable search errors, store error and continue
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    return {"search_results": search_results} # raw search results or error

def bug_tracking(state: EmailAgentState) -> EmailAgentState:
    """Create or update bug tracking ticket"""
    
    # create ticket in bug tracking system
    ticket_id = f"BUG_{uuid.uuid4()}"
    
    return {"ticket_id": ticket_id}

def write_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""

    classification = state.get('classification', {})
 
    # format context from raw state data on demand
    context_sections = []

    if state.get("search_results"):
        # format search results for the prompt
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get("customer_history"):
        # format customer data for the prompt
        context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")

    # build the prompt with formatted context
    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}
    
    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    - Be brief
    """

    response = llm.invoke(draft_prompt)

    # determine if human review is needed based on urgency and intent
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )

    # route to the appropriate next node
    if needs_review:
        print("Needs approval")
        go_to = "human_review"
    else:
        go_to = "send_reply"

    return Command(
        update = {"draft_response": response.content},
        goto = go_to
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get('classification', {})
 
    # interrupt() comes first - any code before it will re-execute on resume
    human_decision = interrupt({
        "email_id": state['email_id'],
        "original_email": state['email_content'],
        "draft_response": state.get('draft_response', ""),
        "urgency": classification.get('urgency', ""),
        "intent": classification.get('intent', ""),
        "action": "Please review and approve/edit this response"
    })

    # process human’s decision
    if human_decision.get("approved"):
        return Command(
        update = {"draft_response": human_decision.get("edited_response", state['draft_response'])},
        goto = "send_reply"
        )
    else:
        # rejection means human will handle directly
        print("Draft rejected. Email not sent.")
        return Command(update = {}, goto = END)  
    
def send_reply(state: EmailAgentState) -> EmailAgentState:
    """Send the email response"""
    # integrate with an email service 
    print(f"Sending reply: {state['draft_response'][:60]}…")
    print("Email sent successfully!")
    return {}      

# Build the graph
builder = StateGraph(EmailAgentState)

# add nodes
builder.add_node("read_email", read_email)
builder.add_node("classify_intent", classify_intent)
builder.add_node("search_documentation", search_documentation)
builder.add_node("bug_tracking", bug_tracking)
builder.add_node("write_response", write_response)
builder.add_node("human_review", human_review)
builder.add_node("send_reply", send_reply)

# add static edges - other edges are generated dynamically
builder.add_edge(START, "read_email")
builder.add_edge("read_email", "classify_intent")
builder.add_edge("classify_intent", "search_documentation")
builder.add_edge("classify_intent", "bug_tracking")
builder.add_edge("search_documentation", "write_response")
builder.add_edge("bug_tracking", "write_response")
builder.add_edge("send_reply", END)

# compile with checkpointer for persistence
memory = InMemorySaver()
app = builder.compile(checkpointer=memory)

# # Visualise graph structure - paste output in https://mermaid.live/
# print(app.get_graph().draw_mermaid()) 

# Example graph invokes

# # Test 1: test with urgent billing issue
# initial_state = {
#     "email_content" : "I was charged twice for my subscription! This is urgent!",
#     "sender_email" : "customer@example.com",
#     "email_id" : "email_123",    
# }

# # run with a thread_id for persistence
# config = {"configurable": {"thread_id": "customer_123"}}
# result = app.invoke(initial_state, config)

# # the graph will pause at human_review and wait for input
# # print(f"Draft ready for review: {result['draft_response'][:60]}…\n")
# print(f"Draft ready for review: {result['draft_response']}…\n")
# decision = input("Approve? (yes/no): ")

# # provide human input to resume 
# human_response = Command(
#     resume = {"approved": decision.lower() == "yes"}
# )

# # resume execution (send or end)
# final_result = app.invoke(human_response, config)


# Test 2: batch test
emails = [
    "I was charged two times for my subscription! This is urgent!",
    "I was wondering if this was available in blue?",
    "Can you tell me how the long sale is on?",
    "The tire won't stay on the car!",
    "My subscription is going to end in a few months, what's the renewal rate?" 
]

needs_approval = [] # list to queue emails that need human review

for i, email_content in enumerate(emails):
    initial_state = {
        "email_content" : email_content,
        "sender_email" : "customer@example.com",
        "email_id" : f"email_{i}",    
    }
    
    print(f"{initial_state['email_id']}: ", end="")
    
    thread_id = str(uuid.uuid4()) # generate unique thread ID for each email
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state, config)
    
    if "__interrupt__" in result:
        result['thread_id'] = thread_id
        needs_approval.append(result)
        
for item in needs_approval:
    interrupt_data = item["__interrupt__"][0].value

    print("\nNeeds approval:")
    print(f"Email ID: {interrupt_data['email_id']}")
    print(f"Original email: {interrupt_data['original_email']}")
    print(f"Draft response: {interrupt_data['draft_response']}")

    decision = input("Approve? (yes/no): ")

    human_response = Command(
        resume={"approved": decision.lower() == "yes"}
    )

    config = {"configurable": {"thread_id": item["thread_id"]}}

    final_result = app.invoke(human_response, config)