package com.example.heart_disease.step4;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class PerformanceMetrics {
    
    /**
     * Generate detailed performance metrics string
     */
    public static String generateMetrics(Evaluation eval, Instances data, String algorithmName) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("\n=== Performance Metrics for ").append(algorithmName).append(" ===\n");
        sb.append(String.format("Accuracy: %.2f%%\n", eval.pctCorrect()));
        sb.append(String.format("Error Rate: %.2f%%\n", eval.pctIncorrect()));
        sb.append(String.format("Kappa Statistic: %.4f\n", eval.kappa()));
        sb.append(String.format("Mean Absolute Error: %.4f\n", eval.meanAbsoluteError()));
        sb.append(String.format("Root Mean Squared Error: %.4f\n", eval.rootMeanSquaredError()));
        sb.append("\n");
        
        sb.append("Class-specific Metrics:\n");
        for (int i = 0; i < data.numClasses(); i++) {
            sb.append("\nClass: ").append(data.classAttribute().value(i)).append("\n");
            sb.append(String.format("  Precision: %.4f\n", eval.precision(i)));
            sb.append(String.format("  Recall: %.4f\n", eval.recall(i)));
            sb.append(String.format("  F-Measure: %.4f\n", eval.fMeasure(i)));
            sb.append(String.format("  ROC Area: %.4f\n", eval.areaUnderROC(i)));
        }
        
        return sb.toString();
    }
}
