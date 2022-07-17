from PIL import Image
import glob
image_list = []
sayac = 0
dogru_sayac = 0
for filename in glob.glob('celebahq/*.png'): #assuming gif
    im=Image.open(filename)
    w, h = im.size
    if (w == 1024 and h == 1024):
        dogru_sayac += 1
    sayac += 1
    print(im.getbands())
    break

print(sayac)
print(sayac - dogru_sayac)
