from enum import IntEnum
from open_scene_graph import OSGSpec

### Prompts ###

### Objects to format prompts for queries ###
class PromptType(IntEnum):
    PLACE_LABELLING = 0
    APPEARANCE_DESCRIPTION = 1
    SCENE_ELEMENT_CLASSIFICATION = 2
    PLACE_LABEL_SIMILARITY = 3
    PAIRWISE_PLACE_MATCHING = 4
    
class PromptStore:

    def __init__(self, spec: type[OSGSpec]):
        self.spec = spec

    def getPrompt(self, prompt_type: type[PromptType]):
        pass
