# COMP9444 Project Evaluation File

# Running python3 eval.py will evaluate each model in the 'models' list 
# according to the metrics mentioned.

from project_requirements import *
from models import model_chooser

models = ["frcnn1", "frcnn2", "frcnn3", "frcnn4"] # = with default weights
#models = ["frcnn1b", "frcnn2b", "frcnn3b", "frcnn4b"] # = without default weights

stat_list = []
for model_name in models:
    model, _ = model_chooser(model_name)
    
    model.load_state_dict(torch.load(DIR + f"/{model_name}.pth"))
    model.to(DEVICE)
    model.eval()
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
        image = Image.open(f).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image).to(DEVICE)
        with torch.no_grad():
            p = model([image])
            for j, bbox in enumerate(p[0]["boxes"]):
                a, b, c, d = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                if p[0]["scores"][j] > 0.66:
                    p_boxes.append(Polygon([(a,b), (a,d), (c,d), (c,b)]))
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
        overlap = ratio * 100
        speed = (end_time - start_time) * 1000
        ratio_list.append(overlap)
        timer_list.append(speed)
        print(f"Model: {model_name}, File: {os.path.basename(f)}, Overlap: {round(overlap,2)}%, Miss Rate: {miss_rate if miss_rate >= 0 else 'None'} Time: {round(speed,2)}ms")

    stat_list.append({"name":model_name,
                      "avg_overlap_pc":round(sum(ratio_list)/len(ratio_list), 2),
                      "avg_miss_rate_pc":round(sum(missr_list)/len(missr_list), 2),
                      "avg_ms_per_image":round(sum(timer_list)/len(timer_list), 2),
                    })

for e in stat_list: print(e)

plot_yolo = True
yolo_overlap = 43.21
yolo_mr = 62.79
yolo_speed = 61.37

colours = ["royalblue"] * len(models)
overlap_bars = [e["avg_overlap_pc"] for e in stat_list]
speed_bars = [e["avg_ms_per_image"] for e in stat_list]
mr_bars = [e["avg_miss_rate_pc"] for e in stat_list]

if plot_yolo:
    models.append("yolov5")
    overlap_bars.append(yolo_overlap)
    speed_bars.append(yolo_speed)
    mr_bars.append(yolo_mr)
    colours.append("lightskyblue")

# Overlap Plot
plt.figure(figsize=(7,5))
plt.ylim(0,100)
plt.bar(models, overlap_bars, color=colours, width=0.5)
plt.minorticks_on()
plt.grid(True, which="both", axis='y', linestyle="--", linewidth=0.5)
for i in range(len(models)):
    plt.annotate(str(overlap_bars[i]), xy=(models[i], overlap_bars[i]), ha='center', va='bottom')
plt.title("Comparison of Accuracies (by Overlap)")
plt.ylabel("Average % Overlap with Label")
plt.xlabel("Model")
plt.savefig("eval_overlap.png", format='png')

# Miss Rate Plot
plt.figure(figsize=(7,5))
plt.ylim(0,100)
plt.bar(models, mr_bars, color=colours, width=0.5)
plt.minorticks_on()
plt.grid(True, which="both", axis='y', linestyle="--", linewidth=0.5)
for i in range(len(models)):
    plt.annotate(str(mr_bars[i]), xy=(models[i], mr_bars[i]), ha='center', va='bottom')
plt.title("Comparison of Miss Rates")
plt.ylabel("Average % Miss Rate")
plt.xlabel("Model")
plt.savefig("eval_miss_rate.png", format='png')

# Speed Plot
plt.figure(figsize=(7,5))
plt.bar(models, speed_bars, color=colours, width=0.5)
plt.minorticks_on()
plt.grid(True, which="both", axis='y', linestyle="--", linewidth=0.5)
for i in range(len(models)):
    plt.annotate(str(speed_bars[i]), xy=(models[i], speed_bars[i]), ha='center', va='bottom')
plt.title("Comparison of Inference Speeds")
plt.ylabel("Average Speed per Image (ms)")
plt.xlabel("Model")
plt.savefig("eval_speed.png", format='png')
