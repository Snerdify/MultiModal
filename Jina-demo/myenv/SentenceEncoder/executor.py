# imports 
import torch 
from typing import Optional 
from docarray import DocList , BaseDoc
from docarray.typing import Anytensor
from jina import Executor, requests
from sentence_transformers import SentenceTransformer


# document
class MyBaseDoc(BaseDoc):
    txt:str = ''
    embedding : Optional[Anytensor[5]]= None



class SentenceEncoder(Executor):
    def __init__(self , device: str= 'cpu',  *args , **kwargs):
        super().__init__(*args , **kwargs)
        





    
    @requests
    def foo(self, docs, **kwargs):
        pass
