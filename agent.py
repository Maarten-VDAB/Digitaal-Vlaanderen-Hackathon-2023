"""Define the chat agent."""
"""Load agent."""
from typing import Any, Optional, Sequence

from langchain.schema.messages import SystemMessage
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.agents.loading import AGENT_TO_CLASS, load_agent
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import RAGTool, ContactInfoTool, PDFParserTool
from prompts import SYSTEM_MESSAGE

def get_agent(
    stream_handler: BaseCallbackManager,
    websocket,
    observation,
    retriever,
    language="English",
    memory=None,
    model="gpt-4",
) -> AgentExecutor:
    """Create an LLM agent that can chat and use tools."""
    if not memory:
        memory = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", input_key="input", return_messages=True
        )
    tools = [
        ContactInfoTool(),
        RAGTool(
            retriever=retriever,
            websocket=websocket,
            language=language,
        ),
        PDFParserTool()
    ]
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[stream_handler],
        verbose=True,
        model_name=model,  # gpt-3.5-turbo, gpt-4
        temperature=0,
        request_timeout=270,
    )

    system_message = SystemMessage(content=SYSTEM_MESSAGE.format(language=language))

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs={
            "system_message": system_message,
            "extra_prompt_messages": [
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="stored_observations"),
            ],
        },
        memory=memory,
    )
    return agent, memory

def initialize_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    agent: Optional[AgentType] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    agent_path: Optional[str] = None,
    agent_kwargs: Optional[dict] = None,
    *,
    tags: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Load an agent executor given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        agent: Agent type to use. If None and agent_path is also None, will default to
            AgentType.ZERO_SHOT_REACT_DESCRIPTION.
        callback_manager: CallbackManager to use. Global callback manager is used if
            not provided. Defaults to None.
        agent_path: Path to serialized agent to use.
        agent_kwargs: Additional key word arguments to pass to the underlying agent
        tags: Tags to apply to the traced runs.
        **kwargs: Additional key word arguments passed to the agent executor

    Returns:
        An agent executor
    """
    tags_ = list(tags) if tags else []
    if agent is None and agent_path is None:
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    if agent is not None and agent_path is not None:
        raise ValueError(
            "Both `agent` and `agent_path` are specified, "
            "but at most only one should be."
        )
    if agent is not None:
        if agent not in AGENT_TO_CLASS:
            raise ValueError(
                f"Got unknown agent type: {agent}. "
                f"Valid types are: {AGENT_TO_CLASS.keys()}."
            )
        tags_.append(agent.value if isinstance(agent, AgentType) else agent)
        agent_cls = AGENT_TO_CLASS[agent]
        agent_kwargs = agent_kwargs or {}
        agent_obj = agent_cls.from_llm_and_tools(
            llm, tools, callback_manager=callback_manager, **agent_kwargs
        )
    elif agent_path is not None:
        agent_obj = load_agent(
            agent_path, llm=llm, tools=tools, callback_manager=callback_manager
        )
        try:
            # TODO: Add tags from the serialized object directly.
            tags_.append(agent_obj._agent_type)
        except NotImplementedError:
            pass
    else:
        raise ValueError(
            "Somehow both `agent` and `agent_path` are None, "
            "this should never happen."
        )
    return AgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        callback_manager=callback_manager,
        tags=tags_,
        **kwargs,
    )