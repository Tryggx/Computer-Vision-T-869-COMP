import torch
import cv2
import numpy as np 
# # from models.yolo import Model  # Assuming 'models' is from the YOLOv5 repository

# # Load the model architecture
# model = Model(cfg="C:/Users/trygg/Documents/Master 3.Önn/Computer Vision/Computer-Vision-T-869-COMP/Assignment4/Custom/data.yaml")  # Adjust the path to your model's config file
# Load the trained weights
# model.load_state_dict(torch.load('"C:/Users/trygg/Documents/Master 3.Önn/Computer Vision/Computer-Vision-T-869-COMP/Assignment4/Custom/yolov5n.pt"')['model'])
# model.eval()

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

# Load PyTorch model
model = torch.hub.load('Computer-Vision-T-869-COMP/Assignment4/Custom/yolov5',"custom", path="Computer Vision/Computer-Vision-T-869-COMP/Assignment4/Custom/yolov5n.pt", source="local")
model.eval()  # Set the model to evaluation mode


def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

# def pre_process(input_image):
#     # Resize and normalize the image
#     # Note: You may need to adjust this preprocessing to match your model's requirements
#     input_image = cv2.resize(input_image, (INPUT_WIDTH, INPUT_HEIGHT))
#     input_image = input_image / 255.0  # Normalize
#     input_image = input_image.transpose((2, 0, 1))  # HWC to CHW
#     input_image = torch.tensor(input_image, dtype=torch.float32)
#     input_image = input_image.unsqueeze(0)  # Add batch dimension
#     return input_image
def pre_process(input_image):
    # Resize the image
    input_image = cv2.resize(input_image, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # Normalize the image
    input_image = input_image / 255.0  
    
    # Change the image from HWC to CHW format
    input_image = input_image.transpose((2, 0, 1))  
    
    # Convert the image to a tensor
    input_image = torch.tensor(input_image, dtype=torch.float32)
    
    # Add a batch dimension
    input_image = input_image.unsqueeze(0)  
    
    return input_image

def post_process(input_image, outputs):
      # Lists to hold respective values while unwrapping.
      class_ids = []
      confidences = []
      boxes = []
      # Rows.
      rows = outputs[0].shape[1]
      image_height, image_width = input_image.shape[:2]
      # Resizing factor.
      x_factor = image_width / INPUT_WIDTH
      y_factor =  image_height / INPUT_HEIGHT
      # Iterate through detections.
      print(rows)
      print(outputs[0][0][0])
      for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)


# Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
      indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # Draw bounding box.             
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            # Class label.                      
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])             
            # Draw label.             
            draw_label(input_image, label, left, top)
      return input_image

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break

            # Pre-process the frame
            # Pre-process the frame

        classesFile = "./Computer-Vision-T-869-COMP/Assignment4/Pre-trained/coco.names"
        classes = []
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        input_tensor = pre_process(frame)
        
        # Feed the tensor to the model
        output = model(input_tensor)

        # Post-process the outputs
        processed_outputs = post_process(frame,output)

        # Display the processed frame (you might need to adjust this part)
        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
