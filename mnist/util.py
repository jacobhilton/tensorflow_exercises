def read_labels(filename, number=None):
    with open(filename, "rb") as file:
        file.seek(8)
        if number is None:
            bytes = file.read()
        else:
            bytes = file.read(number)
        labels = map(lambda byte : int(byte.encode("hex"), 16), bytes)
    return labels

def read_images(filename, number=None):
    with open(filename, "rb") as file:
        file.seek(16)
        if number is None:
            bytes = file.read()
        else:
            bytes = file.read(number * 28 * 28)
        pixels = map(lambda byte : int(byte.encode("hex"), 16), bytes)
        images = [[pixels[image_number + row_number * 28:image_number + row_number * 28 + 28] for row_number in xrange(0, 28)] for image_number in xrange(0, len(pixels), 28 * 28)]
    return images


labels = read_labels("t10k-labels-idx1-ubyte")
print labels[0]
print labels[9998]
images = read_images("t10k-images-idx3-ubyte", 5)
print len(images[3])
print len(images[3][12])
import pprint
pprint.pprint(images)
