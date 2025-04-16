from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict

from IPython.display import Image, display
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chat.chatbot.prompts import (CLASSIFICATION_TEMPLATE, CONVERSATION_TEMPLATE, RAG_TEMPLATE, 
                                TRANSFORMATION_TEMPLATE, GUARDRAIL_TEMPLATE, DENIAL_TEMPLATE, 
                                SUMMARIZATION_TEMPLATE)
from utilities.colorize import color, rainbow
from utilities.utilities import docs_to_string, load_config


def amazing_print(text):
    num_of_eq = 40 - len(text) // 2
    text = text.upper()
    print(rainbow("=" * num_of_eq + " " + text + " " + "=" * num_of_eq, True))


def add_messages_with_limit(
    left: List[AnyMessage], right: AnyMessage
) -> List[AnyMessage]:
    size = len(left) + 1
    limit = load_config("./chat/chatbot/config.yaml")["history_size"]
    delta = size - limit
    if delta <= 0:
        return add_messages(left, right)
    else:
        return add_messages(left[delta:], right)


class Router:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CLASSIFICATION_TEMPLATE),
            ]
        )
        self.route_chain = self.prompt | self.llm | JsonOutputParser()

    def invoke(self, inputs):
        return self.route_chain.invoke(inputs)

    async def ainvoke(self, inputs):
        return await self.route_chain.ainvoke(inputs)


def fill_prompt(system_template: str):
    return ChatPromptTemplate.from_messages(
        [("system", system_template), MessagesPlaceholder("history", optional=True)]
    ).with_config(run_name="PromptTemplate")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        input: The input string.
        tool_code: The code of the tool to be executed.
    """

    messages: Annotated[List[AnyMessage], add_messages_with_limit]
    type: Optional[Literal["conversational", "document", "summary"]]
    context: Optional[List[Document]]
    transformed_query: Optional[str]
    is_relevant: Optional[Literal["yes", "no"]]


def print_state(state: GraphState):
    messages = state.get("messages", [])
    msg = "\n"
    for i, m in enumerate(messages):
        msg += type(m).__name__ + ": "
        msg += str(m.content[:100].replace("\n", " ").replace("\t", " "))
        msg += "\n"
    print(color("Messages:", True, "yellow"), msg)

    print(color("Type:", True, "green"), state.get("type", ""))

    context = state.get("context", [])
    ctx = "\n"
    for i, c in enumerate(context):
        ctx += str(i + 1) + ". "
        ctx += str(c.page_content[:100].replace("\n", " ").replace("\t", " "))
        ctx += "\n"
    print(color("Context:", True, "blue"), ctx)
    print(
        color("Transformed Query:", True, "magenta"), state.get("transformed_query", "")
    )
    print(color("Is Relevant:", True, "cyan"), state.get("is_relevant", ""))
    print("\n\n")


class Graph:
    def __init__(self, llm, router, retriever, verbose, config):
        self.llm = llm
        self.router = router
        self.retriever = retriever
        self.verbose = verbose
        self.config = config
        self.handler = config["configurable"].get("handler", None)
        self.transformation_prompt = fill_prompt(TRANSFORMATION_TEMPLATE)
        self.transformation_chain = self.transformation_prompt | llm
        self.conversational_prompt = fill_prompt(CONVERSATION_TEMPLATE)
        self.conversational_chain = self.conversational_prompt | llm
        self.rag_prompt = fill_prompt(RAG_TEMPLATE)
        self.rag_chain = self.rag_prompt | llm
        self.summarization_prompt = fill_prompt(SUMMARIZATION_TEMPLATE)
        self.summarization_chain = self.summarization_prompt | llm
        self.guardrail_prompt = fill_prompt(GUARDRAIL_TEMPLATE)
        self.guardrail_chain = self.guardrail_prompt | llm | JsonOutputParser()
        self.denial_prompt = fill_prompt(DENIAL_TEMPLATE)
        self.denial_chain = self.denial_prompt | llm
        self.compiled = self.build_and_compile()

    async def transform_query(self, state: GraphState):
        if self.verbose:
            amazing_print("Transform Query")
            print_state(state)
        output = await self.transformation_chain.ainvoke(
            {"history": state["messages"][-2:]}
        )
        return {"transformed_query": output.content}

    async def classify_question(self, state: GraphState):
        if self.verbose:
            amazing_print("Classify Question")
            print_state(state)
        response = await self.router.ainvoke({"question": state["messages"][-1]})
        return {
            "type": response.get("type", "conversational"),
            "transformed_query": "",
            "context": [],
            "is_relevant": "",
        }

    async def conversational(self, state: GraphState):
        if self.verbose:
            amazing_print("Conversational")
            print_state(state)
        output = self.conversational_chain.astream({"history": state["messages"]})
        response = await self.stream_output(output)
        return {"messages": AIMessage(content=response)}

    async def rag(self, state: GraphState):
        if self.verbose:
            amazing_print("Rag")
            print_state(state)
        output = self.rag_chain.astream(
            {"history": state["messages"], "context": docs_to_string(state["context"])}
        )
        response = await self.stream_output(output)
        history = self.config["configurable"]["history"]
        history.add_message_from_response(
            {"answer": response, "documents": state["context"]}
        )
        return {"messages": AIMessage(content=response)}

    async def summarization(self, state: GraphState):
        if self.verbose:
            amazing_print("Summarization")
            print_state(state)
        output = self.summarization_chain.astream({"history": state["messages"]})
        response = await self.stream_output(output)
        return {"messages": AIMessage(content=response)}

    def retrieve(self, state: GraphState):
        if self.verbose:
            amazing_print("Retrieve")
            print_state(state)
        if not state.get("transformed_query", ""):
            response = self.get_ctx(state["messages"][-1].content)
        else:
            response = self.get_ctx(state["transformed_query"])
        return {"context": response}

    def has_documents(
        self, state: GraphState
    ) -> Literal["rag", "transformation", "guardrail"]:
        context = state["context"]
        if context:
            return "rag"
        elif not context and not state["transformed_query"]:
            return "transformation"
        else:
            return "guardrail"

    def route(
        self, state: GraphState
    ) -> Literal["retriever", "guardrail", "summarization"]:
        type = state["type"]
        if type == "document":
            return "retriever"
        elif type == "conversational":
            return "guardrail"
        elif type == "summary":
            return "summarization"
        else:
            raise ValueError("Invalid type")

    async def guardrail(self, state: GraphState):
        if self.verbose:
            amazing_print("Guardrail")
            print_state(state)
        output = await self.guardrail_chain.ainvoke(
            {"question": state["messages"][-1].content}
        )
        return {"is_relevant": output.get("is_relevant", "no")}

    async def denial(self, state: GraphState):
        if self.verbose:
            amazing_print("Denial")
            print_state(state)
        output = self.denial_chain.astream({"question": state["messages"][-1].content})
        response = await self.stream_output(output)
        return {"messages": AIMessage(content=response), "type": "denial"}

    def should_respond(self, state: GraphState) -> Literal["denial", "conversational"]:
        if state["is_relevant"] == "yes":
            return "conversational"
        else:
            return "denial"

    async def stream_output(self, output):
        response = ""
        async for o in output:
            response = response + o.content
            await self.handler.on_new_token(o)
        return response

    def get_ctx(self, user_input) -> List[Document]:
        # prendo i documenti che sono stati usati per rispondere alle domande precedenti
        history = self.config["configurable"]["history"]
        follwoup_ctx = history.get_followup_ctx(
            self.config["configurable"]["followup_threshold"]
        )

        # prendo i documenti che sono simili alla domanda dell'utente
        docs = self.retriever.invoke(user_input, followup_docs=follwoup_ctx)
        return docs

    def build_and_compile(self):
        """Builds and compiles the graph."""
        workflow = StateGraph(GraphState)

        workflow.add_node("transformation", self.transform_query)
        workflow.add_node("classification", self.classify_question)
        workflow.add_node("conversational", self.conversational)
        workflow.add_node("rag", self.rag)
        workflow.add_node("summarization", self.summarization)
        workflow.add_node("retriever", self.retrieve)
        workflow.add_node("guardrail", self.guardrail)
        workflow.add_node("denial", self.denial)

        workflow.add_edge(START, "classification")
        workflow.add_edge("conversational", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("summarization", END)
        workflow.add_edge("transformation", "retriever")
        workflow.add_edge("denial", END)

        workflow.add_conditional_edges("retriever", self.has_documents)
        workflow.add_conditional_edges("classification", self.route)
        workflow.add_conditional_edges("guardrail", self.should_respond)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory, debug=False)

    def print(self, show=False):
        """Prints the graph structure and saves it to a .png file."""
        try:
            # Get the image object
            img = self.compiled.get_graph(xray=True).draw_mermaid_png()
            # Save to file
            with open("graph.png", "wb") as f:
                f.write(img)
            # Also display in notebook if possible
            if show:
                display(Image(img))
        except Exception:
            # This requires some extra dependencies and is optional
            pass


class App:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.config = graph.config
        self.handler = graph.handler
        self.compiled = graph.compiled

    async def run(self, input: dict, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            history = self.config["configurable"]["history"]
            history.add_message_from_user(input)
            response = await self.compiled.ainvoke(input, self.graph.config)
            if self.handler:
                await self.handler.end()
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return {}
