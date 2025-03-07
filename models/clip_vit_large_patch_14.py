from PIL import Image
import requests
import cv2
import math
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
vidObj = cv2.VideoCapture('/home/ha25/ByborgAI/data/sceneclipautoautotrain00004.avi')
fps = int(vidObj.get(cv2.CAP_PROP_FPS))
total_frame = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

per_every_seconds = 1

highest_similarity_score = 0
most_similar_frame = 0

for current_frame in range(0, total_frame, per_every_seconds * fps):
    # current_frame = i * per_every_seconds * fps
    vidObj.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    _, frame = vidObj.read()
    cv2.imwrite(f"./models/frames/{current_frame}.jpg", frame)

    image = Image.open(f"./models/frames/{current_frame}.jpg")
    inputs = processor(text=["two mens and a dog"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    similarity_score = logits_per_image[0][0].item()
    if similarity_score > highest_similarity_score:
        print('high')
        most_similar_frame = current_frame
        highest_similarity_score = similarity_score

    print(similarity_score)
    # print(probs)


print(f'most similar frame is: {most_similar_frame}')


