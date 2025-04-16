import asyncio

import httpx
import numpy as np
import pyrubberband as pyrb
import soundfile as sf
import tiktoken

from utilities.colorize import color
from utilities.tts_utilities import (AudioFragment, MultipleTextRequest,
                                     TextFragment, TextRequest)


class Manager:
    def __init__(self, n_max_threads):
        self.n_threads = 0
        self.n_max_threads = n_max_threads

    def is_busy(self):
        return self.n_threads >= self.n_max_threads

    def add_threads(self, n):
        if not self.is_busy() and self.n_threads + n <= self.n_max_threads:
            self.n_threads += n
            print("Active threads:", self.n_threads)
            print("Free threads:", self.get_free_threads())
            return True
        return False

    def remove_threads(self, n):
        if self.n_threads - n >= 0:
            self.n_threads -= n
            print("Active threads:", self.n_threads)
            print("Free threads:", self.get_free_threads())
            return True
        return False

    def get_free_threads(self) -> int:
        return self.n_max_threads - self.n_threads

    def is_working(self) -> bool:
        return self.n_threads > 0


class AudioBuffer:
    """Gestisce il buffering di testi e frammenti audio."""

    def __init__(self, manager: Manager, max_tokens=200):
        self.text_fragments = []  # lista dei testi in arrivo
        self.audio_fragments = []  # lista dei frammenti audio
        self.lock = asyncio.Lock()
        self.manager = manager
        self.max_tokens = max_tokens
        print(
            color("[AUDIO BUFFER]", True, "magenta"),
            ": Audio buffer initialized",
            sep="",
        )

    async def add_text(self, text: TextFragment):
        """Aggiunge un testo al buffer."""
        async with self.lock:
            self.text_fragments.append(text)

    async def add_audio(self, audio: AudioFragment):
        """Aggiunge un frammento audio al buffer."""
        async with self.lock:
            self.audio_fragments.append(audio)

    async def _get_audio(self) -> np.ndarray | None:
        """Restituisce l'audio completo concatenando i frammenti."""
        async with self.lock:
            sorted_fragments = sorted(
                self.audio_fragments, key=lambda x: (x.id, x.sub_id)
            )
            if sorted_fragments:
                audio = np.concatenate([fragment.content for fragment in sorted_fragments])
                return audio
            return None

    async def save_audio(self, path: str):
        """Salva l'audio completo su file."""
        audio = await self._get_audio()
        stretched = pyrb.time_stretch(audio, 22050, 1.1)
        sf.write(path, stretched, 22050, format="wav")
        await self.clear()
        print(
            color("[AUDIO BUFFER]", True, "magenta"), ": Audio saved to ", path, sep=""
        )

    async def is_complete(self):
        """Verifica se il maker ha finito di generare tutti i frammenti audio."""
        async with self.lock:
            return len(self.text_fragments) == 0 and not self.manager.is_working()

    async def clear(self):
        """Resetta il buffer."""
        async with self.lock:
            self.text_fragments = []
            self.audio_fragments = []

    async def send(self):
        """Invia fino a self.limit TextRequest al TTS per la generazione dell'audio."""
        async with self.lock:
            if not self.manager.is_busy():
                try:
                    items = [
                        self.text_fragments.pop(0)
                        for _ in range(
                            min(
                                self.manager.get_free_threads(),
                                len(self.text_fragments),
                            )
                        )
                    ]
                    if items:
                        requests = [TextRequest(text=r.text, id=r.id) for r in items]
                    else:
                        return
                    data = MultipleTextRequest(requests=requests)
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://localhost:9000/", json=data.model_dump()
                        )
                        self.manager.add_threads(len(requests))
                        if response.json().get("status", "error") == "error":
                            raise Exception(
                                f"Errore {response.status_code} durante la richiesta al TTS"
                            )
                        print(
                            color("[AUDIO BUFFER]", True, "magenta"),
                            ": Request sent to TTS (",
                            len(requests),
                            " fragments)",
                            sep="",
                        )
                except Exception as e:
                    print(color("[AUDIO BUFFER]", True, "red"), ": Error:", e)
            else:
                print(
                    color("[AUDIO BUFFER]", True, "magenta"),
                    ": TTS is busy, waiting for free threads",
                    sep="",
                )

    def split_text_into_chunks(self, text, encoding="cl100k_base"):
        """Divide il testo in segmenti rispettando il limite massimo di token."""
        tokenizer = tiktoken.get_encoding(encoding)
        tokens = tokenizer.encode(text)

        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i: i + self.max_tokens]
            chunks.append(tokenizer.decode(chunk_tokens))
        if len(chunks) > 1:
            print(
                color("[AUDIO BUFFER]", True, "magenta"),
                ": Text split into",
                len(chunks),
                "chunks",
            )
        return chunks
