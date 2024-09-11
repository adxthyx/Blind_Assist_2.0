import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera index

# Inference function
@torch.no_grad()
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.get_image_features(**inputs)
    captions = model.get_image_to_text_similarity(outputs, text_input_ids=None)
    caption = processor.decode(captions.argmax(dim=-1)[0], skip_special_tokens=True)
    return caption

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was captured successfully
    if ret:
        # Convert the OpenCV frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Generate caption
        caption = generate_caption(image)

        # Display the resulting frame and caption
        cv2.imshow('frame', frame)
        print(f"Caption: {caption}")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()