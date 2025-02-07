import datatypes
import uuid

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


llm = None

summarize_instruction = """You are an expert in {topic}. Your task is to write a **structured summary** of the following text:

{text}

### Key Instructions:

1. **Thoroughness**: Cover every important detail, ensuring completeness.
2. **Structured by Subtopics**: Rather than a single large summary, break it down into multiple subtopics, each with its own summary.
3. **Adjustments**: Follow these specific suggestions (if provided): {adjustments}

Return the response in the required structured format.
"""


def summarize(state: datatypes.ChapterInState):
    # in this I do wanna create a list of summeries about each different subtopic in the text.
    structured_llm = llm.with_structured_output(datatypes.SummariesStructure)
    instruction = summarize_instruction.format(topic=state.topic, text=state.text, adjustments=state.adjustments)
    response = structured_llm.invoke(instruction)
    print(response)
    return {"summaries": response.summaries}

def validate(state: datatypes.ChapterState):
    print(state)
    # in this i want to check if the summaries did capture all information
    pass

def continue_to_fact_creation(state: datatypes.ChapterState):
    pass

def create_facts(state: datatypes.ChapterState):
    # this node should creaet a facts for the summaries
    pass

def validate_facts(state: datatypes.ChapterState):
    # this node should check if facts span all important areas of this 
    pass

def create_report(state: datatypes.ChapterState):
    # this node should create a conceise report of the summaries of the input
    pass


def create_graph():
    builder = StateGraph(datatypes.ChapterState, input=datatypes.ChapterInState)

    builder.add_node("summarize", summarize)
    builder.add_node("validate", validate)
    
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "validate")
    builder.add_edge("validate", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    img_bytes = graph.get_graph().draw_mermaid_png()
    img = mpimg.imread(io.BytesIO(img_bytes), format="png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    return graph

if __name__ == "__main__": 
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    graph = create_graph()
    
    text = """Mainz, the capital of Rhineland-Palatinate in western Germany, is a city rich in history, culture, and tradition. Located on the banks of the Rhine River, it is known for its picturesque old town, vibrant university life, and as the birthplace of Johannes Gutenberg, the inventor of the printing press.

    One of Mainz’s most striking features is its historical architecture. The Mainz Cathedral (Mainzer Dom), an impressive Romanesque structure with Gothic and Baroque elements, dominates the cityscape. Nearby, the Gutenberg Museum pays tribute to the city’s most famous son and showcases original prints of the Gutenberg Bible.

    The city’s old town is filled with charming half-timbered houses, bustling squares, and lively markets. Marktplatz and Kirschgarten are particularly popular spots, offering a taste of traditional Mainz with their cozy cafés and wine taverns. Speaking of wine, Mainz is a key city in the Rheinhessen wine region, Germany’s largest wine-producing area, making it a paradise for wine enthusiasts. The annual Mainzer Weinmarkt (Wine Market) and various wine festivals highlight the city's strong connection to viticulture.

    Mainz is also famous for its carnival, or "Fastnacht", one of Germany’s biggest and most colorful celebrations. Every year, thousands gather to enjoy parades, satirical performances, and the city’s unique carnival culture, which is deeply rooted in humor and political satire.

    With its blend of historical significance, lively cultural scene, and scenic location along the Rhine, Mainz offers a unique experience that captures the essence of German tradition and modern vibrancy. Whether you’re interested in history, wine, or simply enjoying a relaxed atmosphere, Mainz has something to offer for everyone."""

    graph_input = datatypes.ChapterInState(topic="mainz", text=text)
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    graph.invoke(graph_input, config)
