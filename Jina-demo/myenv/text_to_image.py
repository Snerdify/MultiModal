import numpy as np
# in jina when you build a model or service it is always in the form a executor
# requests is used to handle request sent to flow
from jina import Executor , requests
# Basedoc class is the fundamental building block of docarray which represents the 
# a single document or a datapoint
# anydocarray is an array of documents
# Doclist is a implementation of docarry and it is a python list of Basedocs
from docarray import BaseDoc , Doclist

# imagedoc is derive from basedoc and is a document for handling images 
# imagedoc can contain imagetensor , imageurl , anyembedding , imagebytes 
from docarray.documents import ImageDoc


# Document , EXECUTOR 


# basedoc is the baseclass for all documents , 
# this class is subclassed to create a new document
class ImagePrompt(BaseDoc):
    text:str

class TextToImage(Executor):
    # this is the constructor of the the class
    # it is called when you create an instance of the class

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # the pipeline is used to generate images
        from diffusers import StableDiffusionPipeline
        import torch
        # the pipeline is configured to use CUDA device for computation

        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype = torch.float16).to("cuda")

# generate_images is a request handler method
    @requests
    # self is a reference to instance of texttoimage executor class
    # the 2nd parameter indicates a list of documents of type imageprompts
    # -> specifies the return type of the function
    # Doclist[Imagedoc] indicates that function is supposed to return a list of documents of type imagedoc
    def generate_images(self, docs:Doclist[ImagePrompt],**kwargs)-> Doclist[ImageDoc]:
        # below line generates images based on text , using a pretrained model pipeline
        # takes a list of text prompts from the docs , pass them through a model
        # and obtain a list of images
        images = self.pipe(docs.text).images 
        for i, doc in enumerate(docs):
            doc.tensor = np.array(images[i])



        





