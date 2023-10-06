# to see how would we pass the device that we want the executor to use

from typing import Optional
from jina import Deployment
from docarray import DocList, BaseDoc
from docarray.typing import AnyTensor
from executor import SentenceEncoder


class MyDoc(BaseDoc):
    text: str = ''
    embedding: Optional[AnyTensor[5]] = None

# here we are trying to demonstrate the use of the encoder by encoding 10000 text documents

def generate_docs():
    for _ in range(10000):
        yield MyDoc(
            text= 'Uisng GPU allows us to speed up encoding'
        )

dep = Deployment(uses = SentenceEncoder , uses_with={'device':'cpu'})

with dep:
    dep.post(on='/', inputs = generate_docs , show_progress= True , request_size = 32 , return_type= DocList[MyDoc])

