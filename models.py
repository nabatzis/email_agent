from typing import TypedDict
from typing_extensions import Annotated

from langgraph.graph import add_messages

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
    
    