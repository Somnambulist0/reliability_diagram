import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os

label_to_name = {
    0: 'person',
    1: 'rider',
    2: 'car',
    3: 'truck',
    4: 'bus',
    5: 'train',
    6: 'motorcycle',
    7: 'bicycle'
}
name_to_label = {v: k for k, v in label_to_name.items()}
json_id_to_pkl_label = {
    24: name_to_label['person'],
    25: name_to_label['rider'],
    26: name_to_label['car'],
    27: name_to_label['truck'],
    28: name_to_label['bus'],
    31: name_to_label['train'],
    32: name_to_label['motorcycle'],
    33: name_to_label['bicycle']
}

def convert_bbox_format(bbox):
    x, y, width, height = bbox
    converted_bbox = [x, y, x + width, y + height]
    return converted_bbox

def load_gt_data(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    gt_data = {}
    for annotation in json_data['annotations']:
        img_id = annotation['image_id']
        if img_id not in gt_data:
            gt_data[img_id] = {'bboxes': [], 'labels': []}
        bbox_converted = convert_bbox_format(annotation['bbox'])
        gt_label_json_id = annotation['category_id']
        gt_label_pkl = json_id_to_pkl_label.get(gt_label_json_id, -1)
        gt_data[img_id]['bboxes'].append(bbox_converted)
        gt_data[img_id]['labels'].append(gt_label_pkl)

    return gt_data

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


buffered_data = []

def print_and_log(output_string, filename='output_pc.txt', buffer_size=100):
    print(output_string, end='')
    buffered_data.append(output_string)

    if len(buffered_data) >= buffer_size:
        with open(filename, 'a') as file:
            file.writelines(buffered_data)
        buffered_data.clear()

def format_output(category, total_preds, correct_preds, precision_values, confidence_bins):
    label_name = label_to_name[category]
    precision_str = ', '.join([f'{pv:.2f}' for pv in precision_values])
    total_preds_str = ', '.join([str(int(tp)) for tp in total_preds])
    correct_preds_str = ', '.join([str(int(cp)) for cp in correct_preds])
    confidence_intervals = ', '.join([f'{confidence_bins[i]:.2f}-{confidence_bins[i+1]:.2f}' for i in range(len(confidence_bins)-1)])

    formatted_string = (f"Category: {label_name}\n"
                        f"Confidence: {confidence_intervals}\n"
                        f"Total Predictions: {total_preds_str}\n"
                        f"Correct Predictions: {correct_preds_str}\n"
                        f"Precision: {precision_str}\n"
                        f"{'-'*30}\n")
    return formatted_string


def main(pkl_dir, json_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_data = load_gt_data(json_dir)

    with open(pkl_dir, "rb") as f:
        results = pickle.load(f)

    confidence_bins = np.arange(0, 1.05, 0.05)
    class_categories = [0, 1, 2, 3, 4, 5, 6, 7]
    class_bin_total_predictions = {category: np.zeros_like(confidence_bins[:-1], dtype=int) for category in class_categories}
    class_bin_correct_predictions = {category: np.zeros_like(confidence_bins[:-1], dtype=int) for category in class_categories}
    for item in results:
        matched_gt_boxes = []
        img_id = item['img_id']
        item['gt_instances'] = gt_data.get(img_id, {'bboxes': [], 'labels': []})
        for score, predicted_box, predicted_label in zip(item['pred_instances']['scores'],
                                                         item['pred_instances']['bboxes'],
                                                         item['pred_instances']['labels']):
            is_correct = False
            predicted_label = predicted_label.item()
            for index, (gt_box, gt_label) in enumerate(
                    zip(item['gt_instances']['bboxes'], item['gt_instances']['labels'])):
                if index not in matched_gt_boxes and IoU(predicted_box, gt_box) > 0.5 and predicted_label == gt_label:
                    is_correct = True
                    matched_gt_boxes.append(index)
                    break

            bin_index = np.digitize(score, confidence_bins) - 1
            class_bin_total_predictions[predicted_label][bin_index] += 1
            if is_correct:
                class_bin_correct_predictions[predicted_label][bin_index] += 1

    output_filename = os.path.join(output_dir, 'output_pc.txt')
    for category in class_categories:
        plt.figure(figsize=(10, 8))
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        precision_values = class_bin_correct_predictions[category] / np.maximum(class_bin_total_predictions[category],
                                                                                1)
        output_string = format_output(category, class_bin_total_predictions[category], class_bin_correct_predictions[category],
                                      precision_values, confidence_bins)

        print_and_log(output_string, output_filename)

        # Draw reliability diagram for this category
        mid_points = [(confidence_bins[i] + confidence_bins[i + 1]) / 2 for i in range(len(confidence_bins) - 1)]
        added_labels = set()

        for i, value in enumerate(precision_values):
            label_below = 'Below Calibration' if 'Below Calibration' not in added_labels else ""
            label_above = 'Above Calibration' if 'Above Calibration' not in added_labels else ""
            label_gap = 'Gap' if 'Gap' not in added_labels else ""

            if value >= mid_points[i]:
                plt.bar(mid_points[i], mid_points[i], width=0.05, color='lightblue', align='center', edgecolor='black',
                        label=label_below)
                added_labels.add('Below Calibration')

                plt.bar(mid_points[i], value - mid_points[i], width=0.05, bottom=mid_points[i], color='lightcoral',
                        align='center', edgecolor='black', label=label_above)
                added_labels.add('Above Calibration')
            else:
                plt.bar(mid_points[i], value, width=0.05, color='lightblue', align='center', edgecolor='black',
                        label=label_below)
                added_labels.add('Below Calibration')

                plt.bar(mid_points[i], mid_points[i] - value, width=0.05, bottom=value, color='lightgray',
                        align='center',
                        edgecolor='black', label=label_gap)
                added_labels.add('Gap')

        plt.plot([0] + mid_points + [1], [0] + mid_points + [1], linestyle='--', color='red',
                 label='Perfect Calibration')
        plt.legend()
        plt.xlabel('Confidence')
        plt.ylabel('Precision')
        plt.title(f'Reliability Diagram for {label_to_name[category]}')
        file_path = os.path.join(output_dir, f'reliability_diagram_{label_to_name[category]}.png')
        plt.savefig(file_path, dpi=300)
        plt.clf()

    if buffered_data:
        with open(output_filename, 'a') as file:
            file.writelines(buffered_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Reliability Diagrams.")
    parser.add_argument('pkl_dir', type=str, help='Path to the .pkl file')
    parser.add_argument('json_dir', type=str, help='Path to the .json file')
    parser.add_argument('output_dir', type=str, nargs='?', default='./reliability_diagram_classes', help='Output directory for the diagrams')
    args = parser.parse_args()
    main(args.pkl_dir, args.json_dir, args.output_dir)