# imports 
import torch 
from typing import Optional 
from docarray import DocList , BaseDoc
from docarray.typing import Anytensor
from jina import Executor, requests
from sentence_transformers import SentenceTransformer


# document
class MyDoc(BaseDoc):
    txt:str = ''
    embedding : Optional[Anytensor[5]]= None



class SentenceEncoder(Executor):
    def __init__(self , device: str= 'cpu',  *args , **kwargs):
        super().__init__(*args , **kwargs)
        self.model = SentenceTransformer('all-MiniLM-L6-v2' , device = device)
        self.model.to(device)






    
    @requests
    def encode( self , docs: DocList[MyDoc] , **kwargs) -> DocList[MyDoc]:
    # add text based embeddings to all documents
        with torch.inference_model():
            embeddings = self.model.encode(docs.texts , batch_size =32) 
            docs.embeddings = embeddings 


