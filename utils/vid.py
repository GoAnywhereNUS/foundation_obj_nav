import cv2
import sys

if __name__ == "__main__":
    vid = cv2.VideoCapture('/dev/video' + sys.argv[1])
    
    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
    
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    vid.release()
