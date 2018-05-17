from PIL import Image
import os.path
import os

class dataAugment(object):
    def __init__(self,dir_out = "G_output/train/",dir_in = "G_input/train/",img_type = "jpg",
                 dir_augin = "augin/",dir_augout = "augout/",box = (60,120,420,480)):
        self.dir_out = dir_out
        self.dir_in = dir_in
        self.dir_augin = dir_augin
        self.dir_augout = dir_augout
        self.img_type = img_type
        self.box = box

    def Origin(self):
        print('origin images...')
        for parent, dirnames, filenames in os.walk(self.dir_in):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im
                newname = self.dir_augin + filename
                out.save(newname)

        for parent, dirnames, filenames in os.walk(self.dir_out):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im
                newname = self.dir_augout + filename
                out.save(newname)

    def HorizontalFlip(self):
        print('horizontal flipping images...')
        for parent, dirnames, filenames in os.walk(self.dir_in):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.FLIP_LEFT_RIGHT)
                newname = self.dir_augin + 'h' + filename
                out.save(newname)

        for parent, dirnames, filenames in os.walk(self.dir_out):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.FLIP_LEFT_RIGHT)
                newname = self.dir_augout + 'h' + filename
                out.save(newname)

    def VerticalFlip(self):
        print('vertical flipping images...')
        for parent, dirnames, filenames in os.walk(self.dir_in):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.FLIP_TOP_BOTTOM)
                newname = self.dir_augin + 'v' + filename
                out.save(newname)

        for parent, dirnames, filenames in os.walk(self.dir_out):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.FLIP_TOP_BOTTOM)
                newname = self.dir_augout + 'v' + filename
                out.save(newname)

    def Rotate90(self):
        print('Rotate 90 degrees images...')
        for parent, dirnames, filenames in os.walk(self.dir_in):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.ROTATE_90)
                newname = self.dir_augin + 'r9' + filename
                out.save(newname)

        for parent, dirnames, filenames in os.walk(self.dir_out):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.ROTATE_90)
                newname = self.dir_augout + 'r9' + filename
                out.save(newname)

    def Rotate180(self):
        print('Rotate 180 degrees images...')
        for parent, dirnames, filenames in os.walk(self.dir_in):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.ROTATE_180)
                newname = self.dir_augin + 'r18' + filename
                out.save(newname)

        for parent, dirnames, filenames in os.walk(self.dir_out):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.ROTATE_180)
                newname = self.dir_augout + 'r18' + filename
                out.save(newname)

    def Rotate270(self):
        print('Rotate 270 degrees images...')
        for parent, dirnames, filenames in os.walk(self.dir_in):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.ROTATE_270)
                newname = self.dir_augin + 'r270' + filename
                out.save(newname)

        for parent, dirnames, filenames in os.walk(self.dir_out):
            for filename in filenames:
                currentPath = os.path.join(parent, filename)
                im = Image.open(currentPath)
                im = im.crop(self.box)
                out = im.transpose(Image.ROTATE_270)
                newname = self.dir_augout + 'r27' + filename
                out.save(newname)

if __name__ == '__main__':
    augmentdata = dataAugment()
    augmentdata.Origin()
    augmentdata.HorizontalFlip()
    augmentdata.VerticalFlip()
    augmentdata.Rotate90()
    augmentdata.Rotate180()
    augmentdata.Rotate270()