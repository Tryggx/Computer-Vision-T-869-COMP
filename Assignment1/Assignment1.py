import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# set timer to measure fps
def builtIn():
    while(True):
        timer = cv2.getTickCount()  
        #### FPS counter 
        ret, frame = cap.read()
        # calculate fps
        # latency = (cv2.getTickCount() - timer) / cv2.getTickFrequency()*1000


        original = frame.copy()
        blur = 15
        #blur frame 
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)

        # update timer

        ### find the brightest spot 
        # find the brightest spot in the color image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Split the channels: Hue, Saturation, and Value
        h, s, v = cv2.split(hsv)

        # Split the channels: Red, Green, and Blue
        r, g, b = cv2.split(rgb)

        # Subtract the green and blue channels to isolate red
        ultraRed = cv2.subtract(r, g)
        ultraRed = cv2.subtract(ultraRed, b)

        # Find the brightest spot in the Value channel
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(v)

        #find the reddest spot in the R channel
        (minValRed, maxValRed, minLocRed, maxLocRed) = cv2.minMaxLoc(ultraRed)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # draw a circle around the maximum red value
        cv2.circle(original, maxLocRed, 5, (0, 0, 255), 2)

        # draw a circle around the maximum brightness value
        cv2.circle(original, maxLoc, 5, (255, 0, 0), 2)

        #cv2.circle(ultraRed, maxLocRed, 5, (255, 0, 0), 2)
        # put fps on the frame
        cv2.putText(original, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # put the maximum pixel value on the frame
        cv2.putText(original, "Max Brigthness: " + str(maxVal), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(original, "Max Red: " + str(maxValRed), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # cv2.putText(original, "Latency: " + str(latency), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('hsv',ultraRed)
        #print("FPS = ", fps)

        cv2.imshow('frame',original)

        if cv2.waitKey(1) & 0xFF == ord('q'):   
            break

    cap.release()
    cv2.destroyAllWindows()

def myMethod():
    while(True):
        timer = cv2.getTickCount()
        ret, frame = cap.read()

        original = frame.copy()
        frame = frame.astype(np.float32) 
        # Initialize variables for grayscale, brightest spot, and reddest spot
      

        maxVal = 0
        maxLoc = (0, 0)
        maxValRed = 0
        maxLocRed = (0, 0)

        # Iterate over each pixel for grayscale conversion and finding spots
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                B, G, R = frame[i, j]

                # Grayscale conversion using luminance formula
                gray_value = 0.299 * R + 0.587 * G + 0.114 * B
                
                # Update brightest spot
                if gray_value > maxVal:
                    maxVal = gray_value
                    maxLoc = (j, i)  # Swap to (x, y)

                # Calculate red value for reddest spot
                red_value = R - (G + B) / 2

                # Update reddest spot
                if red_value > maxValRed:
                    maxValRed = red_value
                    maxLocRed = (j, i)  # Swap to (x, y)


        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # Drawing and annotations
        cv2.circle(original, maxLoc, 5, (255, 0, 0), 2)    # Brightest spot
        cv2.circle(original, maxLocRed, 5, (0, 0, 255), 2) # Reddest spot

        # put fps on the frame
        cv2.putText(original, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # put the maximum pixel value on the frame
        cv2.putText(original, "Max Brigthness: " + str(maxVal), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(original, "Max Red: " + str(maxValRed), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        #cv2.imshow('hsv',ultraRed)
        #print("FPS = ", fps)

        cv2.imshow('frame',original)

        if cv2.waitKey(1) & 0xFF == ord('q'):   
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #builtIn()
    myMethod()
