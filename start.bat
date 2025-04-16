@echo off
start cmd /k "python audio_buffer_main.py"
start cmd /k "python audio_maker_main.py"
streamlit run chatbot_main.py