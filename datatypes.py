import uuid

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing import List, Annotated
from typing_extensions import TypedDict
from operator import add
from reducers import add_reducer

class ChapterInState(BaseModel):
    topic: str = Field(description="topic of the section")
    text: str = Field(description="text to summarize")
    adjustments: str = Field(description="suggested adjustments", default="")

class SummaryStructure(BaseModel):
    topic: str = Field(description="Subtopic of the summary")
    summary: str = Field(description="Detailed summary for the given subtopic")

class SummariesStructure(BaseModel):
    summaries: List[SummaryStructure] = Field(description="list of all summaries")

class ChapterState(TypedDict):
    topic: str
    report: str
    summaries: List[str]
    facts: Annotated[List[str], add]

class FactState(TypedDict):
    topic: str
    fact: str

class QAPair(BaseModel):
    id: str = Field(
        description="unique identifyer for the question",
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str = Field(
        description="question to ask"
    )
    result: str = Field(
        description="the information the question is based on"
    )
    answer: str = Field(
        description="answer by the user"
    )
    # think about adding the validation as a datafield

class Questions(MessagesState):
    num_questions: int = Field(
        description="number of questions"
    )
    current_question: int = Field(
        description="index of current question",
    )
    topic: str = Field(description="topic of the facts")
    questions: Annotated[List[QAPair], add_reducer] = Field(
        description="list of Qusetions with their corresponding awnsers"
    )

class QuizInState(MessagesState):
    topic: str = Field(description="topic of the facts")
    num_questions: int = Field(description="number of questions", default=0)
    facts: List[str] = Field(description="a fact a question can be formed around")
    current_question: int = Field(description="current question index", default = 0)
