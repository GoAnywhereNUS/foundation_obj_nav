import torch

from open_scene_graph import (
    OpenSceneGraph,
    OSGSpec,
    default_scene_graph_specs
)
from prompts import PromptStore, PromptType

class OSGMapper:
    """
    A program sketch of an OSG mapping algorithm.
    Specifies the algorithm and prompts in templated
    form, using placeholder spatial concept from the 
    OSG meta-structure.

    Takes as input an OSG spec, which allows it to be
    fully instantiated as an executable routine with
    concrete prompts to LLMs/VLMs.
    """
    def __init__(self, spec_str=default_scene_graph_specs):
        """
        Input: spec, JSON string
        """
        self.OSG = OpenSceneGraph(spec_str)
        self.spec = self.OSG.getSpec()
        self.prompt_store = PromptStore(self.spec)

    def parseImage(self, imgs):
        for view, img in imgs.items():
            place_label = self.query_vqa(
                img, self.prompt_store.getPrompt(PromptType.PLACE_LABELLING)
            )

    def estimateState(self):
        pass

    def updateOSG(self):
        pass