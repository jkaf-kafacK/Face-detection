import ast
import csv
import xml.etree.ElementTree as ET
import numpy as np
with open("data_Testing_celeba.csv") as file:
    keys = ['name_image', 'box_image', 'coord_point_image']
    ready = csv.DictReader(file, fieldnames=keys)
    table_data = np.array([])
    resutls = [ ]
    for row in ready:
        table_data = np.append(table_data,dict(row))
        #resutls.append(dict(row))
    #resutls
    table_data
#print(table_data)
# root = xml.Element("datasettrain")
# tree = xml.ElementTree(root)
# createElement = xml.Element("name")
# createElement.text = "celebaMask-HQ-img face point dataset - training images"
# root.append(createElement)
# createElement1 = xml.Element("comment", )
# createElement1.text = "The dataset is actually a " \
#                       "combination of the 62_celebaMask-HQ-img, 63_celebaMask-HQ-img, and 64_celebaMask-HQ-img face " \
#                       "landmark datasets.  But the iBUG people have aggregated it all together and gave them a " \
#                       "consistent set of 68 landmarks across all the images Finally, note that the bounding boxes are " \
#                       "from dlib's default face detector.  For the faces the detector failed to detect, we guessed at " \
#                       "what the bounding box would have been had the detector found it and used that "
#
# root.append(createElement1)
# create_images = xml.Element("images")
# root.append(create_images)

tree = ET.parse("labels_ibug_300W_and_celeba1300_test.xml")
root = tree.getroot()

number_file = 0
for elm in root: # open file xml
    if elm.tag == 'images':
        for i in range(0, len(table_data)):#open dictionary
            var = table_data[i]
            var_name = var['name_image']
            var_box = var['box_image']
            var_coord = var['coord_point_image']
            list_box = ast.literal_eval(var_box)
            list_convet_tuple_matrix = ast.literal_eval(var_coord)
            number_file = number_file + 1
            print("number esecuted : {}".format(number_file))
            createSubImage = ET.SubElement(elm, 'image', attrib={'file': str(var_name)})
            createBox = ET.SubElement(createSubImage, 'box',
                                        attrib={'top': str(list_box[0]), 'left': str(list_box[1]), 'height': str(list_box[2]),
                                                'width': str(list_box[3])})
            for p, point in enumerate(list_convet_tuple_matrix):
                createPart = ET.SubElement(createBox, 'part',
                                            attrib={'name': str(p).zfill(2), 'x': str(point[0]), 'y': str(point[1])})
            with open("labels_ibug_300W_and_celeba1300_test.xml", 'wb') as f:
                tree.write(f)

