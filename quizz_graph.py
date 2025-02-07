import datatypes
import random
import uuid

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from typing import Literal
### PROMTS ###
question_instruction = """You are a Professor at a University teaching about {topic}. Your goal is to create an exam question about the following information: "{fact}" in which the given information serves as the result of the question. Pay attention about the following details: 1: The Answer should not be given in the question
2: The Question should be somewhat open, meaning no multiple choice
3: The Question should ask about the essential information of the given fact.
4: Return just the question."""

validation_instruction = """You are an expert in {topic} and correcting a test by a student. The question the student had to answer was:

{question}

The answer given by the student was:

{answer}

The expected answer is: {result}

Validate if the given answer is correct. Respond in one of 3 ways:

1. If the answer by the student is correct, respond with: 'continuing with next question' only.
2. If the answer is incorrect, provide a **brief** explanation of why it is wrong and what the correct answer should be.
3. If the student's response **explicitly** indicates that they want to stop the quiz (e.g., "I want to stop," "I don't want to continue," "end the quiz"), respond with: 'quiz ended on user behalf' only.  
   - **Do not** assume the student wants to stop just because they answer "no" or provide an incorrect response.  
   - Only use this option if the student **clearly emphasizes** their intent to stop.
"""

answer_instruction = """You are an expert in {topic}. Regarding the following exam question:

{question}

The user gave the following incorrect answer: {answer}, while the correct answer is: {result}. 

Now, the user has asked about it. It is your role to provide a satisfactory response. Keep these key points in mind:

1. Give a **short and concise** explanation.
2. Stick **strictly** to the question, topic, and answer.
3. If the user **did not ask a question**, respond only with: 'Glad I could help'.
4. If the user **explicitly expresses** that they want to stop (e.g., "I want to stop," "I don’t want to continue," "end the quiz"), respond only with: 'quiz ended on user behalf'.  
   - **Do not assume** the user wants to stop just because they answer "no" or provide an incorrect response.  
   - Only use this response if the user **clearly emphasizes** their intent to stop.
"""
##############

llm = None

### NODES NEEDED ###
def gen_questions(state: datatypes.QuizInState):
    facts = state["facts"]
    #llm = ChatOpenAI(model="gpt-4o", temperature=0)
    facts_to_ask = random.sample(facts, state["num_questions"])
    # somehow handle the picking of questions that the user performed badly on
    return {"facts": facts_to_ask}


def continue_to_question_gen(state: datatypes.QuizInState):
    return[Send("gen_question", {"topic": state["topic"], "fact": fact}) for fact in state["facts"]]

def gen_question(state: datatypes.FactState):
    prompt = question_instruction.format(topic=state["topic"], fact=state["fact"])
    response = llm.invoke(prompt)
    return {"questions": [datatypes.QAPair(question=response.content, result=state["fact"], answer="")]}

def ask_question(state: datatypes.Questions):
    question_idx = state["current_question"]
    question = state["questions"][question_idx].question
    print(question)
    return {"messages": [AIMessage(content=question)]}

def human_answer(state: datatypes.Questions):
    """ No-opt for human answer interaction """
    pass

# need an edge to decide weather to go there 
def should_continue_with_quiz(state: datatypes.Questions) -> Literal["ask_question", "give_context", "uptate_knowledge"]:
    message = state["messages"][-1].content
    if "continuing with next question" in message.lower():
        if state["num_questions"] - 1 >= state["current_question"]:
            print("\n\n")
            return "cleanup"
        else: 
            return END
    if "quiz ended on user behalf" in message.lower():
        return END
    print(f"\n\n{message}\n\nDo you have any questions about this?")
    return "ask_for_further_question"

def ask_for_further_question(state: datatypes.Questions):
    return {"messages": [AIMessage(content="Any further Questions?")]}

def validate_res(state: datatypes.Questions):
    # break fefore to wait for answer
    human_answer = state["messages"][-1].content
    question=state["questions"][state["current_question"]]
    answer = question.result
    prompt = validation_instruction.format(topic=state["topic"], question=question.question, answer=human_answer, result=answer)
    response=llm.invoke(prompt)
    id = question.id
    return {"messages": [response], "questions": [datatypes.QAPair(id=id, question=question.question, result=question.result, answer=human_answer)]}
    

def give_context(state: datatypes.Questions): 
    # break before that node
    feedback = state["messages"][-1].content
    question = state["questions"][state["current_question"]]
    instruction = answer_instruction.format(topic=state["topic"], question=question.question, answer=question.answer, result=question.result)
    response = llm.invoke([SystemMessage(content=instruction)]+state["messages"])
    return {"messages": [response]}

def should_continue_after_questions(state: datatypes.Questions):
    message = state["messages"][-1].content
    if "glad i could help" in message.lower(): 
        if state["num_questions"] - 1 >= state["current_question"]:
            print("\n\n")
            return "cleanup"
        else:
            return END
    
    if "quiz ended on user behalf" in message.lower():
        return END
    print(f"\n\n{message}\nAny further questions?\n")
    return "ask_for_further_question"

def cleanup(state: datatypes.Questions):
    # return a state with all messages mark for deletion and the current question idx ++
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
    return {"messages": delete_messages, "current_question": state["current_question"] + 1}

def update_knowledge():
    pass
####################

def generate_graph():
    builder = StateGraph(datatypes.Questions, input=datatypes.QuizInState)

    builder.add_node("gen_questions", gen_questions)
    builder.add_node("gen_question", gen_question)
    builder.add_node("ask_question", ask_question)
    builder.add_node("validate_res", validate_res)
    builder.add_node("give_context", give_context)
    builder.add_node("cleanup", cleanup)
    builder.add_node("ask_for_further_question", ask_for_further_question)

    builder.add_edge(START, "gen_questions")
    builder.add_conditional_edges("gen_questions", continue_to_question_gen, ["gen_question"])
    builder.add_edge("gen_question", "ask_question")
    builder.add_edge("ask_question", "validate_res")
    builder.add_conditional_edges("validate_res", should_continue_with_quiz, [END, "ask_for_further_question", "cleanup"])
    builder.add_edge("ask_for_further_question", "give_context")
    builder.add_conditional_edges("give_context", should_continue_after_questions, [END, "ask_for_further_question", "cleanup"])
    builder.add_edge("cleanup", "ask_question")
    
    memory = MemorySaver();

    graph = builder.compile(interrupt_before=["validate_res", "give_context"], checkpointer=memory)
    img_bytes = graph.get_graph().draw_mermaid_png()
    img = mpimg.imread(io.BytesIO(img_bytes), format="png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    return graph

if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    graph = generate_graph()
    thread = {"configurable": {"thread_id": uuid.uuid4()}}
    initial_input = {"current_question": 0,"topic": "new york city", "num_questions": 4, "facts": [
    "New York City has the most skyscrapers in the U.S., with over 300 buildings.",
    "Central Park is 843 acres, making it larger than the country of Monaco.",
    "The Statue of Liberty was a gift from France in 1886 as a symbol of freedom.",
    "The NYC subway has 472 stations and operates 24/7.",
    "For decades, the price of a pizza slice and a subway ride were nearly the same, known as the 'Pizza Principle'."
    ]}
    user_input = "fist"
    while user_input != "end":
        user_input = "end"
        for e in graph.stream(initial_input, thread):
            continue
        initial_input = None
        state = graph.get_state(thread)
        if len(state.next) == 0:
            break
        user_input = input()
        graph.update_state(
            thread,
            {"messages": [HumanMessage(content=user_input)]}
        )

    

