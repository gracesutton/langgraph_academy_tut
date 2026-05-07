# LangGraph State Machine — Human-in-the-Loop Demo

# This script demonstrates core LangGraph concepts without using an LLM.
# In a real agent, the hardcoded routing logic in node_a would be replaced by an LLM call. The graph structure, memory, and interrupt patterns would remain identical.

# Concepts demonstrated:
#   - State: a TypedDict with an operator.add reducer to append (not overwrite)
#   - Nodes: functions that read state and return updates
#   - Routing: node_a uses Command to control its own routing (vs. conditional edges)
#   - Memory: InMemorySaver checkpoints state after every step, persisting it across turns
#   - Human-in-the-loop: interrupt() pauses the graph and waits for human input, then resumes via Command(resume=...)
#   - Thread ID: identifies the conversation session in the checkpointer, reusing the same thread ID gives the graph memory across turns

import operator
from typing import Annotated, List, Literal, TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# Memory
memory = InMemorySaver()  # instantiate a checkpointer using InMemorySaver()
config = {"configurable": {"thread_id": "1"}} # configure a thread ID with arbitrary thread ID of 1

# State class definition
class State(TypedDict):
    nlist : Annotated[List[str], operator.add] # reducer function to append to state (not overwrite)

# Node functions
# Node a reads state, decides the next node, and updates state.
def node_a(state: State) -> Command[Literal["b", "c", END]]:  # node controls its own routing.
   print("Entered 'a' node")
   select = state["nlist"][-1] # take the last value that was written to state
   if select == "b": # if b —> branch to node b
      next_node = "b"
   elif select == "c": # if c —> branch to node c
      next_node = "c"
   elif select == "q": # if q —> branch to node end
      next_node = END
   else: # else call human-in-the-loop with LangGraph interrupt function
      admin = interrupt(f"Unexpected input '{select}'")  # returns user input
      print(admin)
      if admin == "continue":
            next_node = "b"
      else:
            next_node = END
            select = "q"

   return Command(goto = [next_node]) # goes to next node

def node_b(state: State) -> State:
   return (State(nlist = ["B"])) # node b will add its label to state

def node_c(state: State) -> State:
   return (State(nlist = ["C"])) # node c will also add its label to state

# instantiate a StateGraph object with our State class
builder = StateGraph(State)

# add nodes
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)

# add edges
builder.add_edge(START, "a")
builder.add_edge("b", END)
builder.add_edge("c", END)

# compile the graph
graph = builder.compile(checkpointer=memory) # include checkpointer when compiling the graph

# chat loop to take users input, print input, and invoke the graph using the input
while True:
    user = input('b, c, or q to quit: ') # take users input
    input_state = State(nlist = [user])
    result = graph.invoke(input_state, config)  # invoke the graph using the input, with memory
    print(result)

    # interrupt handler
    if '__interrupt__' in result:
        print(f"Interrupt:{result}")
        msg = result['__interrupt__'][-1].value # extract the interrupt message from the list
        print(msg)
        human = input(f"\n{msg}: ") # prompt human for input

        human_response = Command(resume = human) # resume property set to human response
        result = graph.invoke(human_response, config)  # re-invoke the graph w/ same memory
 
    # q to quit
    if result['nlist'][-1] == "q":
        print("quit")
        break