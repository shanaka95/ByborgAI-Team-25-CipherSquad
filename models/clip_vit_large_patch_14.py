from PIL import Image
import requests
import cv2
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
vidObj = cv2.VideoCapture('/home/ha25/ByborgAI/data/sceneclipautoautotrain00001.avi')
fps = int(vidObj.get(cv2.CAP_PROP_FPS))
total_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))



# success = True
# count = 0

# while(success):
#     success, frame = vidObj.read()
#     cv2.imwrite(f'./models/frames/{count}.jpg', frame)
#     count+=1

for i in range(0, total_frame, fps):
    total_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


image = Image.open("./models/frames/0.jpg")
inputs = processor(text=["a photo of a girl", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

print(logits_per_image)
print(probs)
