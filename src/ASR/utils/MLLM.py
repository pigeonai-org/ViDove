from abc import ABC, abstractmethod

class MLLMAPI(ABC):
    @abstractmethod
    def describe_image(self, image):
        pass