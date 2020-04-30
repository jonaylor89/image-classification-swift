
import Foundation
import PythonKit
let np = Python.import("numpy")

let file = "dataset.csv"
let np_dataset = np.loadtxt(file, delimiter: ",")

let dataset: [Vector] = Array(np_dataset)!

var total_avg: Float = 0
for k in 1..<6 {

    let scores = evaluate(on: dataset, k: k, n_folds: 10)
    let sum = scores.reduce(0, +)
    let length = Float(scores.count)
    let average = sum / length

    print("\n")
    print("[INFO] k=\(k)")
    print("\tScores: \(scores)")
    print("\tMean Accuracy: \(average)%")

    total_avg += average
}

print("\n[INFO] Total Average Accuracy \(total_avg / 5)%")

