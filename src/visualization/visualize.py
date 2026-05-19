import cv2
from src.utils.helpers import load_image


def visualize_detections(images_path, labels_path, image_name, image_extension):

    image = load_image(f"{images_path}/{image_name}.{image_extension}")
    with open(f"{labels_path}/{image_name}.txt", "r") as f:
        label_lines = f.readlines()

    for line in label_lines:
        class_id, x_center, y_center, width, height = map(float, line.split())

        # Choose the color according to the class id (red for fire, blue for smoke)
        if class_id == 0:
            color = (255, 0, 0)  # Red
        else:
            color = (0, 0, 255)  # Blue

        # Convert from YOLO format to pixel coordinates
        img_height, img_width, _ = image.shape
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # Display the image with detections
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_predictions(image, pred_boxes):

    print(pred_boxes)

    for cls, x_center, y_center, width, height, score in pred_boxes:

        # Choose the color according to the class id (red for fire, blue for smoke)
        if cls == 0:
            color = (255, 0, 0)  # Red
        else:
            color = (0, 0, 255)  # Blue

        # Convert from YOLO format to pixel coordinates
        img_height, img_width, _ = image.shape
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

        # Write the confidence score in the rectangle
        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Display the image with detections
    cv2.imshow("Predictions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_path = "data/dataset/raw/test/images"
    labels_path = "data/dataset/raw/test/labels"
    image_name = "WEB10751"
    image_extension = "jpg"
    # image_name = "PublicDataset01056"
    visualize_detections(images_path, labels_path, image_name, image_extension)
