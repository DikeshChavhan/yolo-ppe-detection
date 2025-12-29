import cv2

def draw_results(results):
    img = results.plot()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
