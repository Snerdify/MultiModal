from jina import Client
from docarray import BaseDoc , DocList
from docarray.documents import ImageDoc


class ImagePrompt(BaseDoc):
    text:str

image_text = ImagePrompt(text='rainbow unicorn butterfly kitten')

client = Client(port=8000)
response = client.post(on='/' , inputs= DocList[ImagePrompt]([image_text]), return_type = DocList[ImageDoc])

# display the response
response[0].display()


# notes on jina client and jina deployment
# recommended to run deployment before client ,
# bcoz when u start the deployment file , you essentially start your jina flow
# jina deployment sets up the executors , drivers , routers that jina client interacts with 
