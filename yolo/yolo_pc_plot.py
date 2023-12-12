import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
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

def process_data(pkl_file, output_folder):
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)

    confidence_bins = np.arange(0, 1.05, 0.05)
    bin_total_predictions = np.zeros_like(confidence_bins[:-1], dtype=int)
    bin_correct_predictions = np.zeros_like(confidence_bins[:-1], dtype=int)

    for item in results:
        matched_gt_boxes = []
        for score, predicted_box, predicted_label in zip(item['pred_instances']['scores'], item['pred_instances']['bboxes'], item['pred_instances']['labels']):
            is_correct = False
            for index, (gt_box, gt_label) in enumerate(zip(item['gt_instances']['bboxes'], item['gt_instances']['labels'])):
                if index not in matched_gt_boxes and IoU(predicted_box, gt_box) > 0.5 and predicted_label == gt_label:
                    is_correct = True
                    matched_gt_boxes.append(index)
                    break

            # put into corresponding confidence bin
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

    #plot
    plt.figure(figsize=(10, 8))
    bar_width = 0.05
    mid_points = [(confidence_bins[i] + confidence_bins[i+1]) / 2 for i in range(len(confidence_bins)-1)]

    # perfect_calibration_line，from(0, 0)to(1, 1)
    perfect_calibration_x = [0] + mid_points + [1]
    perfect_calibration_y = [0] + mid_points + [1]
    plt.plot(perfect_calibration_x, perfect_calibration_y, linestyle='--', color='red', label='Perfect')

    # bar
    plt.bar(0, 0, color='lightblue', label='Below')
    plt.bar(0, 0, color='lightsalmon', label='Above')
    plt.bar(0, 0, color='lightgray', label='Gap')
    plt.legend(fontsize=30)

    for i, mid_point in enumerate(mid_points):
        if mid_point > precision_values[i]:
            plt.bar(mid_point, precision_values[i], width=bar_width, color='lightblue', align='center', edgecolor='black')
            plt.bar(mid_point, mid_point - precision_values[i], bottom=precision_values[i], width=bar_width, color='lightgray', align='center', edgecolor='black')
        else:
            plt.bar(mid_point, mid_point, width=bar_width, color='lightblue', align='center', edgecolor='black')
            plt.bar(mid_point, precision_values[i] - mid_point, bottom=mid_point, width=bar_width, color='lightsalmon', align='center', edgecolor='black')

    plt.xlabel('Confidence', fontsize=32)
    plt.ylabel('Precision', fontsize=32)
    plt.tick_params(axis='both', labelsize=30)
    plt.title('Reliability Diagram', fontsize=36)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    output_img_path = os.path.join(output_folder, 'reliability_diagram.png')
    plt.savefig(output_img_path, dpi=300)

def main():
    parser = argparse.ArgumentParser(description="Generate Reliability Diagram.")
    parser.add_argument('pkl_file', type=str, help='Path to the .pkl file')
    parser.add_argument('output_folder', type=str, nargs='?', default='reliability_diagram_result',  help='Output folder for the files')
    args = parser.parse_args()

    process_data(args.pkl_file, args.output_folder)

if __name__ == "__main__":
    main()
