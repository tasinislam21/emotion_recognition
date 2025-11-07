import cv2
from timm import create_model
import torch
import torchvision
import torchvision.transforms as transforms

device = 'cuda'
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=2)  # Using a pretrained ViT model
model = model.to(device)
model.load_state_dict(torch.load('best.pt', map_location=device))
model.eval()

cv2.namedWindow("emotion detector")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

font = cv2.FONT_HERSHEY_SIMPLEX
org = (150, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

while rval:
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (224, 224))
    tensorImg = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensorImg = (tensorImg - mean) / std
    tensorImg = tensorImg.unsqueeze(0)
    tensorImg = tensorImg.to(device)
    with torch.no_grad():
        outputs = model(tensorImg)
    _, predicted = torch.max(outputs.data, 1)
    id2word = {0:"Negative", 1:"Positive"}
    text = id2word.get(predicted.cpu().tolist()[0])
    frame = cv2.putText(frame, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()