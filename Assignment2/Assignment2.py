import cv2
import numpy as np
import random



def ransac_line_fitting(edges, num_iterations=200, threshold=5, k=10, proximity=100, linearity_threshold=0.8):
    best_line = None
    best_support = 0
    best_inliers = []

    y_coords, x_coords = edges
    edge_points = list(zip(x_coords, y_coords))

    if len(edge_points) // k < 2:
        return None, []

    for _ in range(num_iterations):
        idx = random.sample(range(0, len(edge_points), k), 2)
        points = [edge_points[i] for i in idx]
        
        # print(np.linalg.norm(np.subtract(points[0], points[1])))
        # if np.linalg.norm(np.subtract(points[0], points[1])) > proximity:
        #     continue

        (x1, y1), (x2, y2) = points
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            inliers = []
            for x, y in edge_points:
                if abs((m * x + b) - y) < threshold:
                    inliers.append((x, y))

            if len(inliers) > best_support:
                best_support = len(inliers)
                best_line = (m, b)
                best_inliers = inliers


    return best_line, best_inliers

cap = cv2.VideoCapture(0)

while(True):
    time = cv2.getTickCount()
    ret, frame = cap.read()

    # Use a different edge detector or fine-tune Canny
    edges = cv2.Canny(frame, 400, 600)

    edge_points = np.where(edges == 255)
    best_line, inliers = ransac_line_fitting(edge_points, num_iterations=200, threshold=5, k=10, proximity=50)

    if best_line is not None:
        m, b = best_line
        x1, y1 = 0, int(b)
        x2, y2 = frame.shape[1], int(m * frame.shape[1] + b)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x, y in inliers:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)


    frame_resized_back = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    time2 = cv2.getTickCount()
    calc_time = (time2 - time) / cv2.getTickFrequency() * 1000
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - time)
    cv2.putText(frame_resized_back, "Latency: " + str(int(calc_time)) + " ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame_resized_back, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow('edges', edges)
    cv2.imshow('frame', frame_resized_back)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
