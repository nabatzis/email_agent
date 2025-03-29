import os

from dotenv import load_dotenv

from models import State

from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from typing import List, Dict, Literal
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from IPython.display import Image, display


from agent_prompts import (
    create_prompt,
    TRIAGE_SYSTEM_PROMPT, 
    TRIAGE_USER_PROMPT, 
    prompt_instructions
)
from memory import (
    AGENT_SYSTEM_PROMPT_MEMORY,
    getStore,
    manage_memory_tool,
    search_memory_tool,
)
from test_data import get_test_email
from tools import *
from triage_router import triage_router

_ = load_dotenv()


profile = {
    "name": "Niko",
    "full_name": "Nikolaos Abatzis",
    "user_profile_background": "Senior software engineer leading a team of 4 developers",
}




import io
import base64
from PIL import Image as PImage

def display_image_bytes_in_terminal(image_bytes):
    """Displays an image in the terminal from bytes, using iTerm2's inline image protocol."""
    try:
        image = PImage.open(io.BytesIO(image_bytes))
        
        # Resize image if necessary
        max_width = 256
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), PImage.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # iTerm2 inline image escape sequence
        print(f"\x1b]1337;File=inline=1;preserveAspectRatio=1:{img_str}\x07")

    except Exception as e:
        print(f"Error displaying image: {e}")

# Example usage:
# Assuming you have image_data as bytes
# display_image_bytes_in_terminal(image_data)


def main():
    print("Create an InMemoryStore ...")
    # From docs, Semantic search is disabled by default. To enable it
    # we need to provide an index.
    store = InMemoryStore(
        index={"embed": "openai:text-embedding-3-small"}
    )
    
   

    print("-----------------------------------------------------------")
    print("Initializing system prompt using user profile, triage rules")
    system_prompt = TRIAGE_SYSTEM_PROMPT.format(
        full_name=profile["full_name"],
        name=profile["name"],
        examples=None,
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
    )


    # print("Create user prompt using the email we want to triage")
    # email = get_test_email()
    # user_prompt = TRIAGE_USER_PROMPT.format(
    #     author=email["from"],
    #     to=email["to"],
    #     subject=email["subject"],
    #     email_thread=email["body"],
    # )

    # print("Invoke the Router LLM with a system prompt and a user prompt(email we want to triage)")
    # result = llm_router.invoke(
    #     [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ]
    # )

    # print(result)

    tools = [
        write_email,
        schedule_meeting,
        check_calendar_availability,
        manage_memory_tool,
        search_memory_tool,
    ]

    # Create the react (reason - act) agent
    response_agent = create_react_agent(
        "anthropic:claude-3-5-sonnet-latest",
        tools=tools,
        prompt=create_prompt,
        # Use this to ensure the store is passed to the agent
        store=getStore(),
    )
    
    email_agent = StateGraph(State)
    email_agent = email_agent.add_node(triage_router)
    email_agent = email_agent.add_node("response_agent", response_agent)
    email_agent = email_agent.add_edge(START, "triage_router")
    email_agent = email_agent.compile(store=store)

    # display(Image(email_agent.get_graph(xray=True).draw_mermaid_png()))
    
    display_image_bytes_in_terminal(email_agent.get_graph(xray=True).draw_mermaid_png(output_file_path='email_agent.png'))
   
    
    
    config = {"configurable": {"langgraph_user_id": "niko_a", "profile": profile}}
    
    # ----------------------------------------------------------
    #  Test of storing info in the store and using for prompring    
    # ----------------------------------------------------------
    # print("sending message to store saying 'Jim is my friend'")
    # response = response_agent.invoke(
    #     {"messages": [{"role": "user", "content": "Jim is my friend"}]},
    #     config=config
    # )

    # for m in response["messages"]:
    #     m.pretty_print()

    # print("Testing response agent with question about 'Jim'")        
    # response = response_agent.invoke(
    #     {"messages": [{"role": "user", "content": "who is jim?"}]},
    #     config=config
    # )        
    
    # for m in response["messages"]:
    #     m.pretty_print()
    # ----------------------------------------------------------

    # ----------------------------------------------------------    
    #  Current behavior
    # ----------------------------------------------------------    
    email_input = {
            "author": "Alice Jones <alice.jones@bar.com>",
            "to": "Niko Abatzis <nikolaos.abatis@nouss.com>",
            "subject": "Quick question about API documentation",
            "email_thread": """Hi Niko,

        Urgent issue - your service is down. Is there a reason why?""",
        }
    print(f"Processing email: {email_input['email_thread']}\n\n")
    response = email_agent.invoke(
        { "email_input": email_input },
        config=config
    )
    
    for m in response["messages"]:
        m.pretty_print()
    # ----------------------------------------------------------    
    
        
if __name__ == "__main__":
    main()
