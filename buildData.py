from PIL import Image
import numpy
import cv2


def add_pendant(x, y, frame, pendant):
	frame = numpy.array(frame)
	frame_img = Image.fromarray(frame)
	pendant = pendant.convert('RGBA')
	frame_img.paste(pendant, (x, y), mask=pendant.split()[3])
	frame = numpy.asarray(frame_img)
	return frame


def get_rect(rects):
	pen_x = 0
	pen_y = 0
	pen_w = 0
	pen_h = 0
	for i, (x, y, w, h) in enumerate(rects):
		pen_y = y
		pen_h = h
		if not i:
			pen_x = x
			pen_w -= x
		else:
			if x > pen_x:
				pen_w += (x + w)
			else:
				pen_x = x
				pen_w = -pen_w - x + w
	return pen_x, pen_y, pen_w, pen_h


def check_data(eye_x, eye_y, eye_w, eye_h, old_data):
	scale_factor = 0.5
	if (abs(eye_x - old_data[0]) > scale_factor * old_data[0] or abs(eye_y - old_data[1]) > scale_factor * old_data[1] or abs(eye_w - old_data[2]) >= scale_factor * old_data[2] or abs(eye_h - old_data[3]) >= scale_factor * old_data[3]):
		return old_data[0], old_data[1], old_data[2], old_data[3]
	return eye_x, eye_y, eye_w, eye_h


def qqvideo():
	detector = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")

	cap = cv2.VideoCapture(0)
	old_data = None
	while True:
		_, frame = cap.read()
		pendant = Image.open("4.png")

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

		pen_x, pen_y, pen_w, pen_h = get_rect(rects)
		if old_data is not None:
			pen_x, pen_y, pen_w, pen_h = check_data(pen_x, pen_y, pen_w, pen_h, old_data)
		try:
			pendant = pendant.resize((int(pen_w*1.5), int(pen_h*1.5)))
		except Exception as e:
			pass
		frame = add_pendant(pen_x-int(pen_w*0.25), pen_y-int(pen_h*0.25), frame, pendant)
		old_data = [pen_x, pen_y, pen_w, pen_h]

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	qqvideo()

