import os
from pathlib import Path

import torch
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils

from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab
from model import U2NET  # full size version 173.6 MB


# import torch.optim as optim

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    imo.save(d_dir / (image_name.name.rsplit(".", 1)[0] + '.png'))


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'
    cwd = Path(os.getcwd())
    image_dir = cwd / 'test_data' / 'test_human_images'
    prediction_dir = cwd / 'test_data' / 'test_human_images_results'
    prediction_dir.mkdir(exist_ok=True)
    model_dir = cwd / 'saved_models' / (model_name + '_human_seg') / (model_name + '_human_seg.pth')

    img_name_list = list(image_dir.glob('*'))
    print("Images in test:", len(img_name_list))

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    print("...load U2NET---173.6 MB")
    net = U2NET(3, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_dir, map_location=device)).to(device)
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        image_path = img_name_list[i_test]
        print("inferencing:", image_path.name)

        inputs_test = data_test['image']
        inputs_test = inputs_test.to(next(net.parameters()))
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

            # normalization
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            # save results to test_results folder
            save_output(image_path, pred, prediction_dir)

            del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()
