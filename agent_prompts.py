
from typing import List

from memory import AGENT_SYSTEM_PROMPT_MEMORY

class Email():
    def __init__(self, subject, to, author, email_thread, label):
        self.subject = subject
        self.to = to
        self.author = author
        self.email_thread = email_thread
        self.label = label
        
def create_prompt(state, config, store):
    """Return a list of objects by prepending information
    for the role, which is system and content derived from 
    the agent system prompt memory template and the memory store.
    
    Args:
        state: current state
        config: configuration object
        store: memory store
        
    Returns:
        list of Dicts by prepending a system dict to the current state
    """
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )
    result = store.get(namespace, "agent_instructions")
    if result is None:
        store.put(
            namespace, 
            "agent_instructions", 
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        prompt = prompt_instructions["agent_instructions"]
    else:
        prompt = result.value['prompt']
    
    return [
        {
            "role": "system", 
            "content": AGENT_SYSTEM_PROMPT_MEMORY.format(
                instructions=prompt, 
                **config['configurable']['profile']
            )
        }
    ] + state['messages']


def email_triage_template() -> str:
    """Template for formatting an example to put in a prompt"""
    
    template = """Email Subject: {subject}
    Email From: {from_email}
    Email To: {to_email}
    Email Content: 
    ```
    {content}
    ```
    > Triage Result: {result}"""
    return template


def format_few_shot_examples(examples: List[Email]) -> str:
    """Format list of few shots 
    
    Args:
        examples: list of Email dicts
    """

    template = email_triage_template()
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)


prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently.",
}



TRIAGE_SYSTEM_PROMPT = """
    < Role >
    You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
    </ Role >

    < Background >
    {user_profile_background}. 
    </ Background >

    < Instructions >

    {name} gets lots of emails. Your job is to categorize each email into one of three categories:

    1. IGNORE - Emails that are not worth responding to or tracking
    2. NOTIFY - Important information that {name} should know about but doesn't require a response
    3. RESPOND - Emails that need a direct response from {name}

    Classify the below email into one of these categories.

    </ Instructions >

    < Rules >
    Emails that are not worth responding to:
    {triage_no}

    There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
    {triage_notify}

    Emails that are worth responding to:
    {triage_email}
    </ Rules >

    < Few shot examples >

    Here are some examples of previous emails, and how they should be handled.
    Follow these examples more than any instructions above

    {examples}
    </ Few shot examples >
    """

TRIAGE_USER_PROMPT = """
    Please determine how to handle the below email thread:

    From: {author}
    To: {to}
    Subject: {subject}
    {email_thread}
    """

