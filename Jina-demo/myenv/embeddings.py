# here we use executor class on GPU locally and docker

# imports 
import torch
# library to handle type hints
from typing import Optional 
from docarray import DocList , BaseDoc 
#anytensor is a union of numpy , pytorch , tensorflow tensors

from docarray.typing import AnyTensor
from jina import Executor , requests


# document
class MyDoc(BaseDoc):
    # txt attribute is used to store textual info
    txt:str = ''
    # embeddings are used to store embeddings which are optional
    embedding : Optional[AnyTensor[5]]=None


# executor
class MyGPUExec(Executor):
    def __init__(self, device: str='cpu' , *args,**kwargs):
        super().__init__(*args , **kwargs)
        self.device = device 


# here we are taking MyDocs objects called docs as input and generate random embeddings of shape 5 from these inputs
@requests
def encode(self , docs:DocList[MyDoc],**kwargs) -> DocList[MyDoc]:
    with torch.inference_mode():
        #generate random embeddings
        embeddings = torch.rand((len(docs),5),device=self.device)
        docs.embedding = embeddings
        embedding_device = 'GPU' if embeddings.is_cuda else 'CPU'
        docs.text = [f"Embedings calculated on {embedding_device}"]    

