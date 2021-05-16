from abc import ABC


class ClippedModel(ABC):
    @property
    def agent(self):
        return self._agent