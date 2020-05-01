from yolo_model.yolov3 import YoloV3


from timeit import default_timer as timer
import numpy as np
from PIL import Image
import cv2

def detect_video(model: YoloV3, video_path, output_path=""):

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    frame_idx=0
    while True:
        frame_idx += 1
        if frame_idx %100 != 0:
            continue
        return_value, frame = vid.read()
        image = Image.fromarray(frame)

        image = model.evaluate_pil(image)

        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__=='__main__':
    m = YoloV3()
    detect_video(m, r'/Users/annabaydina/Downloads/Telegram Desktop/IMG_0181.MOV', '/Users/annabaydina/Downloads/Telegram Desktop/output.mp4')
    # m.evaluate(r'f:\Pictures\ItalyNov2019\Rome2019\_DSC9636.jpg')
    # m.evaluate(r'/Users/annabaydina/Downloads/Telegram Desktop/photo_2020-05-01_14-18-08.jpg')
