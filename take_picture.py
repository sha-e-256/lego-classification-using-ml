from gpiozero import Button as RPIButton
from pathlib import Path
from picamera import PiCamera

red_button = RPIButton(21)  # Pin 40 is GPIO21; so use 21
camera = PiCamera()  # Create Raspberry Pi camera object


def take_image():
    save_dir = Path(r'/home/pi/lego-classification-using-ml/testing-images')
    camera.capture(f'{save_dir}/test.jpg')


def main():
    while True:
        red_button.when_pressed = take_image  # Whenever the button is pressed, the current image in the dir is overwritten


if __name__ == '__main__':
    main()