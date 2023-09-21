import model
import torch
from PIL import Image
import requests

# step 1: load the images

img_one = Image.open(requests.get("https://www.google.com/search?sca_esv=567254757&sxsrf=AM9HkKmyAQ6r9aW9Wn5c8JohIzSj28Ficg:1695296584396&q=images&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjyqs71z7uBAxXlZ_UHHXLaB2oQ0pQJegQIDBAB#imgrc=aVgXecnmQ_f1MM", stream=True).raw)

img_two= Image.open(requests.get(
    "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
    stream= True
).raw)


query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
        stream=True
    ).raw
)




'''

 step 2: Process these images

Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
batch_size x num_media x num_frames x channels x height x width. 
In this case batch_size = 1, num_media = 3, num_frames = 1,
channels = 3, height = 224, width = 224.

'''

vision_x=[image_processor(img_one).unsqueeze(0),image_processor(img_two).unsqueeze(0),image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim =0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)


# preprocessing the text

# here we expect an <img> token to indicate where the image is
# we expect <|endofchunk|> to indicate end of text

tokenizer.padding_size = "left" 

