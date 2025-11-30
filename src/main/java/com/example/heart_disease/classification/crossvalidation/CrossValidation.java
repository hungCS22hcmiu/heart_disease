package com.example.heart_disease.classification.crossvalidation;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import com.example.heart_disease.utils.Logger;

import java.util.Random;

public class CrossValidation {

    private Logger logger;

    public CrossValidation(Logger logger) {
        this.logger = logger;
    }

    public Evaluation performCrossValidation(Classifier classifier, Instances data, int folds) throws Exception {
        logger.log("=== " + folds + "-Fold Cross-Validation ===");

        Evaluation eval = new Evaluation(data);
        long cvStartTime = System.currentTimeMillis();
        eval.crossValidateModel(classifier, data, folds, new Random(1));
        long cvTime = System.currentTimeMillis() - cvStartTime;

        logger.log("✓ " + folds + "-fold cross-validation completed!");
        logger.log("✓ Cross-validation time: " + cvTime + " ms");
        logger.log("");

        return eval;
    }

    public void printEvaluationResults(Evaluation eval, Instances data) throws Exception {
        logger.log(eval.toSummaryString());
        logger.log("Confusion Matrix:");
        logger.log(eval.toMatrixString());
        logger.log("");

        logger.log("=== Performance Metrics ===");
        logger.log(String.format("Accuracy: %.2f%%", eval.pctCorrect()));
        logger.log(String.format("Error Rate: %.2f%%", eval.pctIncorrect()));
        logger.log(String.format("Kappa Statistic: %.4f", eval.kappa()));
        logger.log(String.format("Mean Absolute Error: %.4f", eval.meanAbsoluteError()));
        logger.log(String.format("Root Mean Squared Error: %.4f", eval.rootMeanSquaredError()));
        logger.log("");

        logger.log("Class-specific Metrics:");
        for (int i = 0; i < data.numClasses(); i++) {
            logger.log("\nClass: " + data.classAttribute().value(i));
            logger.log(String.format("  Precision: %.4f", eval.precision(i)));
            logger.log(String.format("  Recall: %.4f", eval.recall(i)));
            logger.log(String.format("  F-Measure: %.4f", eval.fMeasure(i)));
            logger.log(String.format("  ROC Area: %.4f", eval.areaUnderROC(i)));
        }
        logger.log("");
    }
}

