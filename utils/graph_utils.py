from dataclasses import dataclass

@dataclass(frozen=True)
class NodeKey:
    node_cls : str
    label : str
    uid : int
    
    def __repr__(self):
        return f'{self.label}_{self.uid}'
    
    def getData(self):
        return (self.node_cls, self.label, self.uid)