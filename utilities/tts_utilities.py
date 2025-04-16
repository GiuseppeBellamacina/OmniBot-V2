import yaml

from typing import List, Optional
from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str
    id: int
    sub_id: Optional[int] = None


class AudioRequest(BaseModel):
    content: List
    id: int
    sub_id: Optional[int] = None


class MultipleTextRequest(BaseModel):
    requests: list[TextRequest]


class MultipleAudioRequest(BaseModel):
    requests: list[AudioRequest]


class AudioFragment:
    """Rappresenta un frammento audio generato dal TTS."""

    def __init__(self, content, id, sub_id=None):
        self.content = content
        self.id = id
        self.sub_id = sub_id

    def __repr__(self):
        return f"AudioFragment {self.id} (len={len(self.content)})"


class TextFragment:
    """Rappresenta un frammento di testo."""

    def __init__(self, text, id, sub_id=None):
        self.text = text
        self.id = id
        self.sub_id = sub_id

    def __repr__(self):
        return f"TextFragment {self.id} (len={len(self.text)})"
