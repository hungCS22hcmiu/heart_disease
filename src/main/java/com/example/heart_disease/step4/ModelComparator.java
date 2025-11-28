package com.example.heart_disease.step4;

import weka.classifiers.Evaluation;

public class ModelComparator {
    
    /**
     * Compare two models and generate comparison report
     */
    public static String compareModels(
            String model1Name, Evaluation eval1, long buildTime1,
            String model2Name, Evaluation eval2, long buildTime2) {
        
        StringBuilder sb = new StringBuilder();
        
        sb.append("\n=== MODEL COMPARISON ===\n\n");
        
        sb.append(String.format("%-30s %-15s %-15s\n", "Metric", model1Name, model2Name));
        sb.append("=".repeat(60)).append("\n");
        
        sb.append(String.format("%-30s %-15.2f %-15.2f\n", "Accuracy (%)", 
                eval1.pctCorrect(), eval2.pctCorrect()));
        sb.append(String.format("%-30s %-15.2f %-15.2f\n", "Error Rate (%)", 
                eval1.pctIncorrect(), eval2.pctIncorrect()));
        sb.append(String.format("%-30s %-15.4f %-15.4f\n", "Kappa Statistic", 
                eval1.kappa(), eval2.kappa()));
        sb.append(String.format("%-30s %-15.4f %-15.4f\n", "Mean Absolute Error", 
                eval1.meanAbsoluteError(), eval2.meanAbsoluteError()));
        sb.append(String.format("%-30s %-15.4f %-15.4f\n", "RMSE", 
                eval1.rootMeanSquaredError(), eval2.rootMeanSquaredError()));
        sb.append(String.format("%-30s %-15d %-15d\n", "Build Time (ms)", 
                buildTime1, buildTime2));
        
        sb.append("\n=== WINNER ANALYSIS ===\n\n");
        
        if (eval1.pctCorrect() > eval2.pctCorrect()) {
            sb.append("Best Accuracy: ").append(model1Name)
              .append(String.format(" (%.2f%%)\n", eval1.pctCorrect()));
        } else {
            sb.append("Best Accuracy: ").append(model2Name)
              .append(String.format(" (%.2f%%)\n", eval2.pctCorrect()));
        }
        
        if (eval1.kappa() > eval2.kappa()) {
            sb.append("Best Kappa: ").append(model1Name)
              .append(String.format(" (%.4f)\n", eval1.kappa()));
        } else {
            sb.append("Best Kappa: ").append(model2Name)
              .append(String.format(" (%.4f)\n", eval2.kappa()));
        }
        
        if (buildTime1 < buildTime2) {
            sb.append("Faster Build: ").append(model1Name)
              .append(String.format(" (%d ms)\n", buildTime1));
        } else {
            sb.append("Faster Build: ").append(model2Name)
              .append(String.format(" (%d ms)\n", buildTime2));
        }
        
        return sb.toString();
    }
}
