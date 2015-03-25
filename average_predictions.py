import numpy as np
import sys

if len(sys.argv) < 3:
    sys.exit("Usage: python average_predictions.py <predictions_file1> [predictions_file_2] [...] <output_file>")

predictions_paths = sys.argv[1:-1]
target_path = sys.argv[-1]
predictions = [np.load(path) for path in predictions_paths]
avg_predictions = np.mean(predictions, axis=0)
np.save(target_path, avg_predictions)
