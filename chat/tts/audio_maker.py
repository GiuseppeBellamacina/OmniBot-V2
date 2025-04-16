import asyncio

import torch
from TTS.api import TTS

from utilities.colorize import color
from utilities.tts_utilities import AudioFragment, TextRequest


class AudioMaker:
    """Gestisce la generazione di audio con TTS."""

    def __init__(self, config):
        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name=self.config["tts_model"]).to(device)
        print(color("[AUDIO MAKER]", True, "cyan"), ": Audio maker initialized", sep="")

    async def generate_audio(self, texts: list[TextRequest]):
        """Genera frammenti audio gestendo memoria GPU in modo efficiente."""
        tasks = []

        for t in texts:
            tasks.append(asyncio.create_task(self._generate_fragment(t)))

        return await asyncio.gather(*tasks)

    async def _generate_fragment(self, t: TextRequest):
        """Genera un singolo frammento audio."""
        print(color("Chunk length:", True, "cyan"), len(t.text))
        try:
            speaker = self.config["speakers"][self.config["speaker_index"]]
            with torch.no_grad():
                fragment = await asyncio.to_thread(
                    self.tts.tts, text=t.text, language="it", speaker=speaker, speed=2.0
                )
            audio_fragment = AudioFragment(content=fragment, id=t.id, sub_id=t.sub_id)
            print(
                color("[AUDIO MAKER]", True, "cyan"),
                f": Audio fragment generated for ID {t.id}{('-' + str(t.sub_id)) if t.sub_id else ''}",
                sep="",
            )
            return audio_fragment
        except Exception as e:
            print(
                color("[AUDIO MAKER]", True, "red"),
                ": Error during audio generation:",
                e,
            )
            torch.cuda.empty_cache()