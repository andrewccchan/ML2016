
from PIL import Image
import sys

if len(sys.argv) != 2 :
    raise Exception('Usage: ./Q2.sh file')

img = Image.open(str(sys.argv[1]))
img = img.rotate(180)
img.save('ans2.png')
