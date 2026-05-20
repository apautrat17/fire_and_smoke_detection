import os
import cv2


def get_random_image_name_in_folder(folder_path):
    import os
    import random

    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    return os.path.join(folder_path, random.choice(image_files))


def predict_image(
    model,
    image_path: str,
    folder_path: str,
    conf_threshold: float = 0.5,
    random: bool = False,
    stop_after_one: bool = True,
):

    if not random:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder {folder_path} does not exist")
        if not os.path.isfile(image_path):
            raise ValueError(f"Image file {image_path} does not exist")
        results = model(image_path, conf=conf_threshold)
        return results[0].show()

    if random and stop_after_one:
        random_image_path = get_random_image_name_in_folder(folder_path)
        results = model(random_image_path, conf=conf_threshold)
        return results[0].show()

    else:
        while True:
            random_image_path = get_random_image_name_in_folder(folder_path)
            print(f"Predicting on image: {random_image_path}")
            results = model(random_image_path, conf=conf_threshold)

            annotated = results[0].plot()

            cv2.imshow("Prediction (q pour quitter)", annotated)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord("q"):
                break
