//
//  KNN.swift
//  
//
//  Created by John Naylor on 4/28/20.
//

import Foundation

func distance(from row1: [Float], to row2: [Float]) -> Float {

    let distanceSq = zip(row1, row2).map {
        $0.0 - $0.1
    }.reduce(0.0, +)

    let dist = sqrt(distanceSq)

    return dist
}

func get_neighbors(_ train: [[Float]], _ test_row: [Float], _ k: Int = 5) -> [[Float]] {
    var distances = train.map {
        ($0, distance(from: Array(test_row[0..<test_row.count-1]), to: Array($0[0..<$0.count-1])))
    }

    distances.sort {
        $0.1 > $1.1
    }

    let neighbors = (1...k).map {
        distances[$0].0
    }

    return neighbors

}

func predict_label(_ train: [[Float]], _ test_row: [Float], _ k: Int = 5) -> Float {

    let neighbors = get_neighbors(train, test_row, k)

    let outputValues = neighbors.map{ $0.last ?? -1 }

    var counts = [Float: Int]()

    outputValues.forEach { counts[$0] = (counts[$0] ?? 0) + 1 }

    let max = counts.max(by: {$0.1 < $1.1})

    return max?.key ?? -1
}

func KNN(train: [[Float]], test: [[Float]], k: Int = 5) -> [Float] {

    let predictions = test.map {
        predict_label(train, $0, k)
    }
    
    return predictions
}


func crossValidationSplit(on dataset: [[Float]], with n_folds: Int) -> [[[Float]]] {

    var datasetSplit = [[[Float]]]()
    var datasetCopy = dataset
    let foldSize = dataset.count / n_folds

    for _ in 0..<n_folds {
        var fold = [[Float]]()

        while fold.count < foldSize {
            let idx = Int.random(in: 1..<datasetCopy.count)
            fold.append(datasetCopy[idx])
            datasetCopy.remove(at: idx)
        }

        datasetSplit.append(fold)
    }
    
    return datasetSplit
    
}

func accuracyMetric(_ actual: [Float], _ predicted: [Float]) -> Float {

    let result = zip(actual, predicted).map() {
        $0.0 == $0.1
    }

    let correctCount = Float(result.filter{ $0 }.count)
    let length = Float(result.count)

    return correctCount / length * 100.0
}

func evaluate(on dataset: [[Float]], k: Int = 5, n_folds: Int = 10) -> [Float] {
    let folds = crossValidationSplit(on: dataset, with: n_folds)
    
    var scores = [Float]()
    
    for (idx, fold) in folds.enumerated() {
        var folds_copy = folds
        
        folds_copy.remove(at: idx)
        
        let trainSet = folds_copy.reduce([], +)
        var testSet = [[Float]]()
        
        for var row in fold {
            row[row.endIndex - 1] = -1
            testSet.append(row)
        }
        
        let predicted = KNN(train: trainSet, test: testSet, k: k)
        let actual = fold.map { $0.last ?? -1 }
        let accuracy = accuracyMetric(actual, predicted)
        
        scores.append(accuracy)
    }
    
    return scores
}
