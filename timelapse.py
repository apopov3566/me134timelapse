import cv2, time, pause, imutils
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def timelapse(interval_secs, max_sim, min_brightness, skip_display=False):
    cam = cv2.VideoCapture(0)

    if not skip_display:
        cv2.namedWindow("test")

    previous = None
    previous_time = int(time.time())

    while True:
        t = time.time()
        t_next = previous_time - (previous_time % interval_secs) + interval_secs
        if skip_display:
            pause.until(t_next)

        ret, frame = cam.read()
        npixels = frame.shape[0] * frame.shape[1]

        if not ret:
            print("failed to grab frame at time: " + str(time.time()))
            break

        brightness = np.linalg.norm(frame) / npixels

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (sim, sim_frame) = (
            (None, frame_grey)
            if previous is None
            else compare_ssim(frame_grey, previous, full=True)
        )

        if not skip_display:
            frame_info = (sim_frame * 255).astype("uint8")
            frame_info = cv2.cvtColor(
                cv2.threshold(
                    frame_info, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1],
                cv2.COLOR_GRAY2BGR,
            )
            frame_info = cv2.putText(
                frame_info,
                str(sim),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            frame_info = cv2.putText(
                frame_info,
                str(brightness),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("test", frame_info)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

        if skip_display or t > t_next:
            previous_time = int(t_next)
            img_name = str(t_next) + ".png"
            outline = (
                "image: "
                + img_name
                + " b: "
                + str(brightness)
                + " d: "
                + str(sim)
                + " | "
            )
            skip = False
            if brightness < min_brightness:
                outline += (
                    "skip b (" + str(brightness) + " < " + str(min_brightness) + ")"
                )
                skip = True
            elif sim is not None and sim > max_sim:
                outline += "skip d (" + str(sim) + " > " + str(max_sim) + ")"
                skip = True
            else:
                outline += "save"

            print(outline)
            if not skip:
                previous = frame_grey
                cv2.imwrite("out/" + img_name, frame)

    cam.release()


if __name__ == "__main__":
    timelapse(1, max_sim=0.95, min_brightness=0.05, skip_display=True)
