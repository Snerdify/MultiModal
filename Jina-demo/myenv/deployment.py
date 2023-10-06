if __name__ == '__main__':

    from text_to_image import TextToImage
    from jina import Deployment


    dep = Deployment(uses= TextToImage , timeout_ready= -1)

    with dep:
            dep.wait_start()