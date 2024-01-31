# COMP9444 Project Inference Code

# Running python3 single_inference.py will run and display the results of two
# input images for the given model in the model_name variable.

from project_requirements import *
from train import class_label
from models import model_chooser

OUTPUT_LIMIT = 1
SHOW = False

model_name = "frcnn1"
model, _ = model_chooser(model_name)

# Load Model
model.load_state_dict(torch.load(f"{DIR}/{model_name}.pth"))
model.to(DEVICE)
model.eval()
TEST_IMAGES = random.sample(glob.glob(IMAGE_PATH + "test" + "/*/*.png"), OUTPUT_LIMIT)
TEST_IMAGES = ['/home/nicholas/Datasets/CityPersons/Cityshapes_images/leftImg8bit/test/munster/munster_000012_000019_leftImg8bit.png']
# Test
for i, f in enumerate(TEST_IMAGES):
    image = Image.open(f).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).to(DEVICE)
    with torch.no_grad():
        p = model([image])
        im = mpimg.imread(f)
        fig, ax = plt.subplots(figsize=(16,9))
        ax.imshow(im)
        table_data = []
        print(f"Test Image Filename: {os.path.basename(f)}")
        for j, bbox in enumerate(p[0]["boxes"]):
            a, b, c, d = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
            if p[0]["scores"][j] > 0.66:
                rect = patches.Rectangle((a, b), c-a, d-b, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                cls = p[0]["labels"][j].item()
                ax.text(a, b, class_label(cls), color='yellow', fontsize='x-large')
                table_data.append([j, a, b, c, d, p[0]['scores'][j].item(), cls, class_label(cls)])
        print(tabulate(table_data, headers=["", "xmin", "ymin", "xmax", "ymax", "confidence", "class", "label"]), "\n")
        plt.savefig(f"./{os.path.basename(f)}_test.png", format='png')
        if SHOW:
            plt.show()
