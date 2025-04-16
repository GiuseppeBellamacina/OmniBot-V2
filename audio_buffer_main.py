import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

from chat.tts.audio_buffer import AudioBuffer, Manager
from utilities.colorize import color
from utilities.utilities import load_config
from utilities.tts_utilities import (AudioFragment, MultipleAudioRequest,
                                    TextFragment, TextRequest)


app = FastAPI()
config = None
buffer = None
manager = None


@app.post("/store_text")  # Viene inviata dall'handler
async def store_text(text: TextRequest):
    """Riceve un testo e avvia la generazione del frammento audio."""
    try:
        data = TextFragment(text.text, text.id)
        chunks = buffer.split_text_into_chunks(data.text)
        for i, c in enumerate(chunks):
            await buffer.add_text(TextFragment(c, text.id, i))
        await buffer.send()
        return {"status": "processing"}
    except Exception as e:
        print(color("[AUDIO BUFFER]", True, "red"), ": Error:", e)
        return {"status": "error", "message": str(e)}


@app.post("/store_audio")  # Viene inviata dal TTS
async def store_audio(audio: MultipleAudioRequest):
    """Riceve un frammento audio e lo aggiunge al buffer."""
    for a in audio.requests:
        try:
            data = AudioFragment(
                np.array([float(x) for x in a.content], dtype=np.float32),
                a.id,
                a.sub_id,
            )
            await buffer.add_audio(data)
            manager.remove_threads(1)
            await buffer.send()
            return {"status": "ok"}
        except Exception as e:
            print(color("[AUDIO BUFFER]", True, "red"), ": Error:", e)
            return {"status": "error", "message": str(e)}


@app.get("/")
async def save_audio_file():
    """Controlla se l'audio è completo e lo salva."""
    try:
        if await buffer.is_complete():
            await buffer.save_audio("tmp.wav")
            torch.cuda.empty_cache()
            return {"status": "ok"}
        else:
            return {"status": "processing"}
    except Exception as e:
        print(color("[AUDIO BUFFER]", True, "red"), ": Error:", e)
        return {"status": "error", "message": str(e)}


@app.get("/start")
def start():
    global config, buffer, manager
    try:
        config = load_config("./chat/tts/config.yaml")
        manager = Manager(config["limit"])
        buffer = AudioBuffer(manager, config["max_tokens"])
        return {"status": "ready"}
    except Exception as e:
        print(color("[AUDIO BUFFER]", True, "red"), ": Error:", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
