from typing import Optional 
from docarray import DocList, BaseDoc
from docarray.typing import AnyTensor
from jina import Deployment
from embeddings import MyGPUExec , MyDoc

dep = Deployment(uses = MyGPUExec , uses_with={'device':'cpu'})

# create a list of Doclist named docs 
# e creates a DocList named docs containing a single instance of the MyDoc class initialized with default values using [MyDoc()]
docs = DocList[MyDoc]([MyDoc()])

with dep:
    # below line sends a post request to encode endpoint with docs as input and we expect the
    # return to be of type DocList containing Mydoc Objects 
    # the result is stored in docs variable
    docs = dep.post(on='/encode' , inputs = docs , return_type = DocList[MyDoc])

# attempts to print embedding attribute of docs
print(f'Document embedding:{docs.embedding}')
# prints text attribute of all docs

print(docs.text)


