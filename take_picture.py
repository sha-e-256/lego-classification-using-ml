#The button mapping, these buttons need to take the pin to ground.
#Also its the GPIOXX number not the actual pin number, e.g. pin 40; GPIO21; use 21
from gpiozero import Button as RPIButton

import time
import os
import subprocess
from pathlib import Path

RedButton = RPIButton(21)

#counter
a=0
# THis works fswebcam -r 320x240 -S 3 --jpeg 50 --save /home/pi/Documents/Capstone2022/Test_Pictures/%H%M%S_0.jpg
SavePicsDirectory = Path('''/home/pi/lego-classification-using-ml/testing-images''')

pic_L = 1280
pic_W = 720

def printdebug ():
    global a
    #print('Button pressed numberS _ '  + str(a))
    print( f"Button pressed number {str(a)}")
    a = a + 1
    
    takePicfswebcam()
    
    
#------------------    
    
def takePicfswebcam():
    
    global pic_L
    global pic_W
    global SavePicsDirectory
    
    #Need the array to work with subprocess call
    command_args = ['fswebcam ',
                    f'-r{pic_L}x{pic_W} ',
                    '-S 3 ',
                    '--no-banner ',
                    '--png 9 ',
                    '--save ',
                    f'{SavePicsDirectory}/%Y%m%d%H%M%S.png']
    
    command = ''.join(command_args)
    
    print('\n\n')
    print ('running: \n' + command)
    subprocess.call(command, shell=True)
    time.sleep(0.25)
    print(f'took a picture number {a}')
    
              
print('Starting..')
print('directory:' ,str(SavePicsDirectory))

while True:
    RedButton.when_pressed = printdebug





