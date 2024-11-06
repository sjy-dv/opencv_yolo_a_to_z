
train:
	- yolo task=detect mode=train data=data.yaml model=yolov8m.pt epochs=20 lr0=0.01


pkg-install:
	- conan profile detect
	- conan profile list
	- conan install . --build=missing