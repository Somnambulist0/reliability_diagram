import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os

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

# Convert bounding box format from [x, y, width, height] to [x1, y1, x2, y2]
def convert_bbox_format(bbox):
    x, y, width, height = bbox
    converted_bbox = [x, y, x + width, y + height]
    return converted_bbox

#load ground truth from annotation JSON file
def load_gt_data(json_file):
    # 从 JSON 文件加载 ground truth 数据
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    gt_data = {}
    for annotation in json_data['annotations']:
        img_id = annotation['image_id']
        if img_id not in gt_data:
            gt_data[img_id] = {'bboxes': [], 'labels': []}
        bbox_converted = convert_bbox_format(annotation['bbox'])
        gt_label = annotation['category_id']
        gt_data[img_id]['bboxes'].append(bbox_converted)
        gt_data[img_id]['labels'].append(gt_label)

    return gt_data

def process_data(pkl_file, json_file, output_folder):
    # Load data
    gt_data = load_gt_data(json_file)
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)

    # data processing
    pkl_to_json_label_mapping = {0: 24, 1: 25, 2: 26, 3: 27, 4: 28, 5: 31, 6: 32, 7: 33}

    confidence_bins = np.arange(0, 1.05, 0.05)
    bin_total_predictions = np.zeros_like(confidence_bins[:-1], dtype=int)
    bin_correct_predictions = np.zeros_like(confidence_bins[:-1], dtype=int)

    for item in results:
        img_id = item['img_id']
        item['gt_instances'] = gt_data.get(img_id, {'bboxes': [], 'labels': []})

        for score, predicted_box, predicted_label in zip(item['pred_instances']['scores'], item['pred_instances']['bboxes'],
                                                         item['pred_instances']['labels']):
            is_correct = False
            predicted_label_json_id = pkl_to_json_label_mapping.get(predicted_label.item(), -1)

            for gt_box, gt_label in zip(item['gt_instances']['bboxes'], item['gt_instances']['labels']):
                if IoU(predicted_box, gt_box) > 0.5 and predicted_label_json_id == gt_label:
                    is_correct = True
                    break

            bin_index = np.digitize(score, confidence_bins) - 1
            bin_total_predictions[bin_index] += 1
            if is_correct:
                bin_correct_predictions[bin_index] += 1

    precision_values = bin_correct_predictions / np.maximum(bin_total_predictions, 1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_text_path = os.path.join(output_folder, 'output.txt')

    with open(output_text_path, 'w') as text_file:
        for i, (low, high) in enumerate(zip(confidence_bins[:-1], confidence_bins[1:])):
            output = f"Confidence Bin: [{low}, {high}): Total Predictions = {bin_total_predictions[i]}, Correct Predictions = {bin_correct_predictions[i]}, Precision = {precision_values[i]:.4f}\n"
            print(output)
            text_file.write(output + '-' * 40 + '\n')

    # plot reliability digram
    plt.figure(figsize=(10, 8))
    bar_width = 0.05
    mid_points = [(confidence_bins[i] + confidence_bins[i + 1]) / 2 for i in range(len(confidence_bins) - 1)]

    perfect_calibration_x = [0] + mid_points + [1]
    perfect_calibration_y = [0] + mid_points + [1]
    plt.plot(perfect_calibration_x, perfect_calibration_y, linestyle='--', color='red', label='Perfect')

    plt.bar(0, 0, color='lightblue', label='Below')
    plt.bar(0, 0, color='lightsalmon', label='Above')
    plt.bar(0, 0, color='lightgray', label='Gap')
    plt.legend(fontsize=30)

    for i, mid_point in enumerate(mid_points):
        if mid_point > precision_values[i]:
            plt.bar(mid_point, precision_values[i], width=bar_width, color='lightblue', align='center',
                    edgecolor='black')
            plt.bar(mid_point, mid_point - precision_values[i], bottom=precision_values[i], width=bar_width,
                    color='lightgray', align='center', edgecolor='black')
        else:
            plt.bar(mid_point, mid_point, width=bar_width, color='lightblue', align='center', edgecolor='black')
            plt.bar(mid_point, precision_values[i] - mid_point, bottom=mid_point, width=bar_width, color='lightsalmon',
                    align='center', edgecolor='black')

    plt.xlabel('Confidence',fontsize=32)
    plt.ylabel('Precision', fontsize=32)
    plt.tick_params(axis='both', labelsize=30)
    plt.title('Reliability Diagram', fontsize=36)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    output_img_path = os.path.join(output_folder, 'reliability_diagram.png')
    plt.savefig(output_img_path, dpi=300)

def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('pkl_file', type=str, help='Path to the .pkl file')
    parser.add_argument('json_file', type=str, help='Path to the .json file')
    parser.add_argument('output_folder', type=str, nargs='?', default='reliability_diagram_result',  help='Output folder for the files')
    args = parser.parse_args()
    process_data(args.pkl_file, args.json_file, args.output_folder)

if __name__ == "__main__":
    main()
