import torch
from src.training.eval import decode_predictions
from src.data.preprocess import letterbox
from src.data.augment import basic_transforms


def load_model_from_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_image(model, image, resize_size, device):
    model.eval()

    with torch.no_grad():
        image = letterbox(image, resize_size=resize_size)[0]

        image = basic_transforms(image).to(device).unsqueeze(0)

        preds = model(image)

        all_boxes = decode_predictions(preds)

        # Reformat from [b, cls, score, x, y, w, h] to [cls, x, y, w, h]
        pred_boxes = [
            [cls, x, y, w, h, score] for b, cls, score, x, y, w, h in all_boxes
        ]

    return pred_boxes
