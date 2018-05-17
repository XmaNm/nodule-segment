from PIL import Image
dir_out = "G_output/train/0091.jpg"
path = "results/"
im = Image.open(dir_out)
# print(im.size)
ro90 = im.transpose(Image.ROTATE_90)
ro180 = im.transpose(Image.ROTATE_180)
ro270 = im.transpose(Image.ROTATE_270)
h = im.transpose(Image.FLIP_LEFT_RIGHT)
v = im.transpose(Image.FLIP_TOP_BOTTOM)
box = (60,120,420,480)
region = im.crop(box)
print(region.size)
# im.show()
region.show()