# COMP9444 Evaluate Pretrained Yolov5 Script

# Running python3 eval_yolo.py will evaluate a pretrained yolov5 model
# according to the metrics mentioned.

from project_requirements import *

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0] # 0 = person, according to the COCO format used by the yolov5 model
model.conf = 0.66 # confidence level threshold
ratio_list = []
timer_list = []
missr_list = []

for f in glob.glob(IMAGE_PATH + "test" + "/*/*.png"):
    p_boxes = [] # predicted boxes
    l_boxes = [] # labelled boxes

    # Get Label Boxes
    label_fpath = f.replace("Cityshapes_images/leftImg8bit", "CityPersons_labels/gtBboxCityPersons").replace(IMAGE_ENDS, LABEL_ENDS)
    with open(label_fpath) as file:
        label = json.load(file)
    for l in label["objects"]:
        if l["label"] in ["pedestrian", "rider"]:
            a, b, c, d = l["bbox"][0], l["bbox"][1], l["bbox"][0]+l["bbox"][2], l["bbox"][1]+l["bbox"][3]
            l_boxes.append(Polygon([(a,b), (a,d), (c,d), (c,b)]))

    # Get Predicted Boxes
    start_time = time.time()
    image = Image.open(f)
    results = model(image)
    r = results.pandas().xyxy[0]
    for i in r.index:
        p_boxes.append(Polygon([
            (r['xmin'][i], r['ymin'][i]),
            (r['xmin'][i], r['ymax'][i]),
            (r['xmax'][i], r['ymax'][i]),
            (r['xmax'][i], r['ymin'][i]),
        ]))
    end_time = time.time()
    
    tp, fp, fn = 0, 0, 0
    if len(p_boxes) == 0 and len(l_boxes) == 0:
        ratio = 1 # If there are no labels, and the model correctly returns no labels.
        miss_rate = 0
    elif len(p_boxes) > 0 and len(l_boxes) == 0:
        ratio = 0 # 100% incorrect.
        miss_rate = 0
    elif len(p_boxes) == 0 and len(l_boxes) > 0:
        ratio = 0
        miss_rate = 1
    else:
        # Calculate % Overlap (Accuracy)
        pred_boxes = gpd.GeoDataFrame(geometry=p_boxes)
        labelled_boxes = gpd.GeoDataFrame(geometry=l_boxes)
        intersect = pred_boxes.overlay(labelled_boxes, how='intersection')
        union = pred_boxes.overlay(labelled_boxes, how='union')
        common_areas = intersect.geometry.area
        total_areas = union.geometry.area
        ratio = sum(common_areas) / sum(total_areas)

        # Calculate Miss Rate, IOU threshold = 0.5
        l_marks = [(l,False) for l in l_boxes] # Polygon types are unhashable :(
        for p in p_boxes:
            o = dict()
            for l in [x[0] for x in l_marks if not x[1]]: # only look at unmarked Ls
                o[p.intersection(l).area / p.union(l).area] = l
            if not o: # If no overlapping labels at all, false positive
                fp += 1
                continue
            l = o[max(o.keys())]
            l_marks[l_marks.index((l,False))] = (l,True) # mark this L
            if l.intersection(p).area / l.union(p).area >= 0.5:
                tp += 1
            else:
                fp += 1
        fn += len([x for x in l_marks if not x[1]])
        try:
            miss_rate = fn / (tp + fn)
        except:
            miss_rate = -1

    if miss_rate >= 0:
        missr_list.append(miss_rate * 100)
    ratio_list.append(ratio*100)
    timer_list.append((end_time - start_time) * 1000)

print(f"avg_overlap_pc: {round(sum(ratio_list)/len(ratio_list), 2)}%")
print(f"avg_miss_rate_pc: {round(sum(missr_list)/len(missr_list), 2)}%")
print(f"avg_ms_per_image: {round(sum(timer_list)/len(timer_list), 2)}ms")
