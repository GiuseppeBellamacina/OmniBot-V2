import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI

from chat.tts.audio_maker import AudioMaker
from utilities.colorize import color
from utilities.utilities import load_config
from utilities.tts_utilities import (AudioRequest, MultipleAudioRequest,
                           MultipleTextRequest)

app = FastAPI()
config = None
maker = None


@app.get("/start")
def start():
    global config, maker
    try:
        config = load_config("./chat/tts/config.yaml")
        maker = AudioMaker(config)
        return {"status": "ready"}
    except Exception as e:
        print(color("[AUDIO MAKER]", True, "red"), ": Error during initialization:", e)
        return {"status": "error", "message": str(e)}


@app.post("/")
async def generate(texts: MultipleTextRequest, background_tasks: BackgroundTasks):
    """Riceve una lista di testi e avvia la generazione di audio."""
    try:
        # Avvia la generazione audio in background
        background_tasks.add_task(process_and_notify, texts.requests)
        return {"status": "processing"}
    except Exception as e:
        print(color("[AUDIO MAKER]", True, "red"), ": Error:", e)
        return {"status": "error", "message": str(e)}


async def process_and_notify(requests):
    """Elabora i testi e invia una notifica al mittente al termine."""
    async with httpx.AsyncClient() as client:
        try:
            results = await maker.generate_audio(requests)
            if results:
                results = [
                    AudioRequest(
                        content=[str(chunk) for chunk in r.content],
                        id=r.id,
                        sub_id=r.sub_id,
                    )
                    for r in results
                ]
            audio_request = MultipleAudioRequest(requests=results)
            # Inviare la POST al mittente
            await client.post(
                config["buffer_url"] + "store_audio", json=audio_request.model_dump()
            )
            print(
                color("[AUDIO MAKER]", True, "cyan"),
                ": Audio fragments sent to buffer",
                sep="",
            )
        except Exception as e:
            error_data = {"status": "error", "message": str(e)}
            await client.post(config["buffer_url"] + "store_audio", json=error_data)
            print(color("[AUDIO MAKER]", True, "red"), ": Error:", e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
