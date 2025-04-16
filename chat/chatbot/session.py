import os

import httpx
import sounddevice as sd
import soundfile as sf
import streamlit as st
from langchain_ollama import ChatOllama

from chat.chatbot.graph import App, Graph, Router
from chat.chatbot.retriever import RetrieverBuilder
from utilities.utilities import ChatHistory, StdOutHandler, load_config
from utilities.colorize import color


class Session:
    def __init__(self, page_title: str, title: str, icon: str, header: str = ""):
        st.set_page_config(page_title=page_title, page_icon=icon)
        st.title(title)
        if header != "":
            st.header(header)

        self.state = st.session_state

    def initialize_session_state(self):
        if "is_initialized" not in self.state or not self.state.is_initialized:
            self.state.is_initialized = False
            self.state.config = load_config("./chat/chatbot/config.yaml")
            print(color("[Session]", True, "green"), ": Config loaded", sep="")

            # Messaggi
            self.state.messages = []
            print(color("[Session]", True, "green"), ": Messages initialized", sep="")

            # History
            self.state.history = ChatHistory(self.state.config["history_size"])
            print(color("[Session]", True, "green"), ": History initialized", sep="")

            # Handler
            self.state.handler = StdOutHandler(
                self.state.config, audio=True, debug=False
            )
            print(color("[Session]", True, "green"), ": Handler initialized", sep="")

            # Retriever
            self.state.retriever = RetrieverBuilder.build(self.state.config)
            if self.state.retriever is None:
                print(
                    color("[Session]", True, "red"),
                    ": Retriever not initialized",
                    sep="",
                )
                return self.state.is_initialized
            print(color("[Session]", True, "green"), ": Retriever initialized", sep="")

            # LLMs
            self.state.llm = ChatOllama(
                model=self.state.config["model"]["name"],
                temperature=self.state.config["model"]["temperature"],
            )
            print(color("[Session]", True, "green"), ": LLM initialized", sep="")

            # Router
            self.state.router = Router(self.state.llm)
            print(color("[Session]", True, "green"), ": Router initialized", sep="")

            # RunnableConfig
            self.state.graph_config = {
                "configurable": {
                    "thread_id": "1",
                    "handler": self.state.handler,
                    "history": self.state.history,
                    "followup_threshold": self.state.config["followup_threshold"],
                },
                "history_size": self.state.config["history_size"],
                "recursion_limit": 15,
            }
            print(
                color("[Session]", True, "green"), ": Graph config initialized", sep=""
            )

            # Graph
            self.state.graph = Graph(
                self.state.llm,
                self.state.router,
                self.state.retriever,
                self.state.config["graph_verbose"],
                self.state.graph_config,
            )
            self.state.app = App(self.state.graph)
            self.state.graph.print()
            print(color("[Session]", True, "green"), ": Graph initialized", sep="")

            # Syncronize with the server
            response = httpx.get("http://localhost:8000/start", timeout=20)
            if response.json().get("status", "error") == "error":
                error = response.json().get("message", "Error")
                self.state.handler.set_audio(False)
                print(error)
                raise Exception(error)
            elif response.json().get("status", "error") == "ready":
                print(
                    color("[Session]", True, "green"),
                    ": AudioBuffer initialized",
                    sep="",
                )

            response = httpx.get("http://localhost:9000/start", timeout=20)
            if response.json().get("status", "error") == "error":
                error = response.json().get("message", "Error")
                self.state.handler.set_audio(False)
                print(error)
                raise Exception(error)
            elif response.json().get("status", "error") == "ready":
                print(
                    color("[Session]", True, "green"),
                    ": AudioMaker initialized",
                    sep="",
                )
  
            self.state.is_generating = False

            self.state.is_initialized = True
            print(color("[Session]", True, "green"), ": Session initialized", sep="")
            return self.state.is_initialized

    
    def disable_input(self):
        self.state.is_generating = True


    async def run(self, prompt: str):
        os.system("cls" if os.name == "nt" else "clear")
        self.state.messages.append({
                "role": "human",
                "content": prompt
        })
        
        with st.chat_message("human"):
            st.markdown(prompt)

        response = None
        input_dict = {"messages": ("user", prompt)}

        with st.chat_message("ai"):
            containers = (st.empty(), st.empty())
            with st.spinner("Elaborazione in corso..."):
                response = await self.state.app.run(input_dict, containers)

        self.state.is_generating = False
        self.state.messages.append({
            "role": "ai",
            "content": response["messages"][-1].content,
            "response_time": self.state.handler.time
        })
        
        st.rerun()


    async def update(self):
        if "is_initialized" not in self.state or not self.state.is_initialized:
            print(color("[Session]", True, "red"), ": Session not initialized", sep="")
            raise Exception(RuntimeError)

        faq_prompt = ""
        with st.sidebar:
            st.markdown("FAQ")
            if st.button("- Qual è l'iter formativo dei piloti in Accademia?", disabled=self.state.is_generating, on_click=self.disable_input):
                faq_prompt = "Qual è l'iter formativo dei piloti in Accademia?"
                
            if st.button("- In cosa consiste la laurea in Medicina e Chirurgia?", disabled=self.state.is_generating, on_click=self.disable_input):
                faq_prompt = "In cosa consiste la laurea in Medicina e Chirurgia?"

            if st.button("- Cosa sai dirmi sui concorsi per gli ufficiali?", disabled=self.state.is_generating, on_click=self.disable_input):
                faq_prompt = "Cosa sai dirmi sui concorsi per gli ufficiali?"

            if st.button("- Come si fa la pasta alla carbonara?", use_container_width=True, disabled=self.state.is_generating, on_click=self.disable_input):
                faq_prompt = "Come si fa la pasta alla carbonara?"

            for _ in range(15):
                st.write("")
            
            if st.button("Clear", use_container_width=True, disabled=self.state.is_generating):
                self.state.messages = []
                self.state.history.clear()
                self.state.graph = Graph(
                    self.state.llm,
                    self.state.router,
                    self.state.retriever,
                    self.state.config["graph_verbose"],
                    self.state.graph_config,
                )
                self.state.app = App(self.state.graph)
                print(color("[Session]", True, "yellow"), ": Session cleared", sep="")
                st.success("Session cleared")

        for message in self.state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "ai":
                    st.markdown(
                        f"⏱ Tempo di risposta: {message['response_time']:.2f} secondi"
                    )

        # AUDIO
        if len(self.state.messages) >= 2:
            if os.path.exists("tmp.wav"):
                cols = st.columns(10)
                with cols[0]:
                    if st.button("🔈", disabled=self.state.is_generating):
                        data, fs = sf.read("tmp.wav", dtype="float32")
                        sd.play(data, fs)
                        sd.wait()
                with cols[1]:
                    if st.button("🔇", disabled=self.state.is_generating):
                        sd.stop()

        print(color("[Session]", True, "magenta"), ": Chatbot ready", sep="")

        if faq_prompt == "":
            if prompt := st.chat_input("Scrivi un messaggio...", key="first_question", disabled=self.state.is_generating, on_submit=self.disable_input):
                await self.run(prompt)
        else:
            st.chat_input("Scrivi un messaggio...", key="fake_input", disabled=True)
            await self.run(faq_prompt)
            faq_prompt = ""
