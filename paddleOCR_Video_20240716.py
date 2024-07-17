import threading
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from paddleocr import PaddleOCR


def draw_text(image, text, position):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font_path = 'M_PLUS_1p/MPLUS1p-Regular.ttf'
    font = ImageFont.truetype(font_path, 20)
    draw.text(position, text, font=font, fill=(0, 0, 255, 0))
    return np.array(img_pil)


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = None
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang='japan')
        self.result_label = tk.Label(window, text="OCR Result")
        self.result_label.pack()
        self.camera_open = False

        # Create buttons
        self.buttons = {
            "Open Camera": self.open_camera,
            "Snapshot": self.take_snapshot,
            "OCR": self.perform_ocr,
            "Exit": self.close
        }

        for button_text, button_command in self.buttons.items():
            tk.Button(window, text=button_text, width=20, command=button_command).pack()

        self.canvas = tk.Canvas(self.window, width=640, height=480)
        self.canvas.pack()

        self.delay = 50  # Adjusted delay for lower fps
        self.update()

    def open_camera(self):
        if not self.camera_open:
            self.vid = cv2.VideoCapture(self.video_source)
            if self.vid.isOpened():
                self.camera_open = True
                self.canvas.config(width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                   height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def take_snapshot(self):
        if self.camera_open:
            ret, frame = self.vid.read()
            if ret:
                self.img_bgr = frame.copy()
                self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.img_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # Display the image in matplotlib for snapshot only
                import matplotlib.pyplot as plt
                plt.figure("Snapshot")
                plt.imshow(self.img_rgb)
                plt.axis('off')
                plt.show()

    def perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):
            self.perform_ocr_thread = threading.Thread(target=self._perform_ocr)
            self.perform_ocr_thread.start()

    def _perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):
            self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('gray_paddle.jpg', self.img_gray)

            result_en = self.ocr_en.ocr(self.img_bgr)
            result_img = self.img_bgr.copy()

            if result_en is not None:
                for line in result_en:
                    for word_info in line:
                        word = word_info[1][0]
                        confidence = word_info[1][1]
                        if confidence > 0.8:
                            coordinates = word_info[0]
                            x_min = int(min(pt[0] for pt in coordinates))
                            y_min = int(min(pt[1] for pt in coordinates))
                            x_max = int(max(pt[0] for pt in coordinates))
                            y_max = int(max(pt[1] for pt in coordinates))
                            cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            result_img = draw_text(result_img, word, (x_min, y_min - 5))

                cv2.imwrite('Result_paddle.jpg', result_img)

                # Extract recognized text
                ocr_result_en = '\n'.join([word_info[1][0] for line in result_en for word_info in line])
                self.result_label.config(text=ocr_result_en)

    def close(self):
        if self.vid and self.vid.isOpened():
            self.vid.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        self.window.quit()

    def draw_text(image, text, position):
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('path/to/mplus-1p-regular.ttf', 20)  # specify the .ttf font file
        draw.text(position, text, font=font, fill=(0, 0, 255, 0))
        return np.array(img_pil)

    def update(self):
        if self.camera_open:
            ret, frame = self.vid.read()
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                result_en = self.ocr_en.ocr(img_bgr)
                result_img = img_bgr.copy()

                if result_en is not None:
                    ocr_result_en = ''
                    for line in result_en:
                        if line is not None:
                            for word_info in line:
                                word = word_info[1][0]
                                print(word)
                                ocr_result_en += word + '\n'
                                confidence = word_info[1][1]
                                if confidence > 0.8:
                                    coordinates = word_info[0]
                                    x_min = int(min(pt[0] for pt in coordinates))
                                    y_min = int(min(pt[1] for pt in coordinates))
                                    x_max = int(max(pt[0] for pt in coordinates))
                                    y_max = int(max(pt[1] for pt in coordinates))
                                    cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                    result_img = draw_text(result_img, word, (x_min, y_min - 5))

                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    # self.result_label.config(text=ocr_result_en)

        self.window.after(self.delay, self.update)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root, "Tkinter and OpenCV")
        root.mainloop()
    except KeyboardInterrupt:
        if app.vid is not None and app.vid.isOpened():
            app.vid.release()
            cv2.destroyAllWindows()  # Close all OpenCV windows
            root.quit()
