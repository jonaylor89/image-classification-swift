//
//  KNN.swift
//  
//
//  Created by John Naylor on 4/28/20.
//

import Foundation

typealias Vector = [Float]

extension Vector {
    func sum() -> Float {
        return self.reduce(0.0, +)
    }

    func distance(from row: Vector) -> Float {

        let distanceSq = zip(self, row).map {
            $0.0 - $0.1
        }.sum()

        let dist = sqrt(distanceSq)

        return dist
    }

    func getNeighbors(_ train: [Vector], _ k: Int = 5) -> [Vector] {
        var distances = train.map {
            (
                $0, 
                Array(self[0..<self.count-1])
                    .distance(from: Array($0[0..<$0.count-1]))
            )
        }

        distances.sort {
            $0.1 > $1.1
        }

        let neighbors = (1...k).map {
            distances[$0].0
        }

        return neighbors
    }

    func maxCount() -> Float {

        var counts = [Float: Int]()

        self.forEach { counts[$0] = (counts[$0] ?? 0) + 1 }

        let max = counts.max (by: {$0.1 < $1.1})

        return max?.key ?? -1
    }

    func predictLabel(_ train: [Vector], _ k: Int = 5) -> Float {

        let neighbors = self.getNeighbors(train, k)

        let outputValues = neighbors.map{ $0.last ?? -1 }

        let prediction = outputValues.maxCount()

        return prediction
        
    }
}


func KNN(train: [Vector], test: [Vector], k: Int = 5) -> Vector {

    let predictions = test.map {
        $0.predictLabel(train, k)
    }
    
    return predictions
}


func crossValidationSplit(on dataset: [Vector], with n_folds: Int) -> [[Vector]] {

    var datasetSplit = [[Vector]]()
    var datasetCopy = dataset
    let foldSize = dataset.count / n_folds

    for _ in 0..<n_folds {
        var fold = [Vector]()

        while fold.count < foldSize {
            let idx = Int.random(in: 1..<datasetCopy.count)
            fold.append(datasetCopy[idx])
            datasetCopy.remove(at: idx)
        }

        datasetSplit.append(fold)
    }
    
    return datasetSplit
    
}

func accuracyMetric(actual: Vector, predicted: Vector) -> Float {

    let result = zip(actual, predicted).map() {
        $0.0 == $0.1
    }

    let correctCount = Float(result.filter{ $0 }.count)
    let length = Float(result.count)

    return correctCount / length * 100.0
}

func evaluate(on dataset: [Vector], k: Int = 5, n_folds: Int = 10) -> Vector {
    let folds = crossValidationSplit(on: dataset, with: n_folds)
    
    var scores = Vector()
    
    for (idx, fold) in folds.enumerated() {
        var folds_copy = folds
        
        folds_copy.remove(at: idx)
        
        let trainSet = folds_copy.reduce([], +)
        var testSet = [Vector]()
        
        for var row in fold {
            row[row.endIndex - 1] = -1
            testSet.append(row)
        }
        
        let predicted = KNN(train: trainSet, test: testSet, k: k)
        let actual = fold.map { $0.last ?? -1 }
        let accuracy = accuracyMetric(actual: actual, predicted: predicted)
        
        scores.append(accuracy)
    }
    
    return scores
}
