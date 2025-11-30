package com.example.heart_disease.classification.evaluation;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import com.example.heart_disease.utils.Logger;
import java.util.Map;

public class ModelEvaluator {

    private Logger logger;

    public ModelEvaluator(Logger logger) {
        this.logger = logger;
    }

    public void printComparisonHeader() {
        logger.log("\n=== PERFORMANCE COMPARISON ===");
        logger.log(String.format("%-25s | %-15s | %-15s", "Metric", "J48", "Random Forest"));
        logger.log("-------------------------------------------------------------------");
    }

    public void compareMetrics(Evaluation eval1, Evaluation eval2, long time1, long time2) {
        logger.log(String.format("%-25s | %-15.2f | %-15.2f", "Accuracy (%)", eval1.pctCorrect(), eval2.pctCorrect()));
        logger.log(String.format("%-25s | %-15.4f | %-15.4f", "Kappa Statistic", eval1.kappa(), eval2.kappa()));
        logger.log(String.format("%-25s | %-15.4f | %-15.4f", "Mean Absolute Error", eval1.meanAbsoluteError(), eval2.meanAbsoluteError()));
        logger.log(String.format("%-25s | %-15.4f | %-15.4f", "Root Mean Squared Error", eval1.rootMeanSquaredError(), eval2.rootMeanSquaredError()));
        logger.log(String.format("%-25s | %-15d | %-15d", "Evaluation Time (ms)", time1, time2));
        logger.log("");
    }

    public void compareClassSpecific(Evaluation eval1, Evaluation eval2, Instances data) throws Exception {
        logger.log("=== Class-specific Comparison ===");
        for (int i = 0; i < data.numClasses(); i++) {
            logger.log("\nClass: " + data.classAttribute().value(i));
            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "Precision", eval1.precision(i), eval2.precision(i)));
            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "Recall", eval1.recall(i), eval2.recall(i)));
            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "F-Measure", eval1.fMeasure(i), eval2.fMeasure(i)));
            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "ROC Area", eval1.areaUnderROC(i), eval2.areaUnderROC(i)));
        }
    }

    public void printConclusion(Evaluation eval1, Evaluation eval2, String model1Name, String model2Name) {
        logger.log("\n=== CONCLUSION ===");
        String betterModel = eval2.pctCorrect() > eval1.pctCorrect() ? model2Name : model1Name;
        double improvement = Math.abs(eval2.pctCorrect() - eval1.pctCorrect());
        logger.log("Better performing model: " + betterModel);
        logger.log(String.format("Accuracy improvement: %.2f%%", improvement));
    }

    public void compareMetricsFromMaps(Map<String, Double> j48Metrics, Map<String, Double> rfMetrics) {
        logger.log(String.format("%-25s | %-15.2f | %-15.2f", "Accuracy (%)",
            j48Metrics.getOrDefault("accuracy", 0.0),
            rfMetrics.getOrDefault("accuracy", 0.0)));
        logger.log(String.format("%-25s | %-15.4f | %-15.4f", "Kappa Statistic",
            j48Metrics.getOrDefault("kappa", 0.0),
            rfMetrics.getOrDefault("kappa", 0.0)));
        logger.log(String.format("%-25s | %-15.4f | %-15.4f", "Mean Absolute Error",
            j48Metrics.getOrDefault("mae", 0.0),
            rfMetrics.getOrDefault("mae", 0.0)));
        logger.log(String.format("%-25s | %-15.4f | %-15.4f", "Root Mean Squared Error",
            j48Metrics.getOrDefault("rmse", 0.0),
            rfMetrics.getOrDefault("rmse", 0.0)));
        logger.log(String.format("%-25s | %-15.0f | %-15.0f", "CV Time (ms)",
            j48Metrics.getOrDefault("cv_time", 0.0),
            rfMetrics.getOrDefault("cv_time", 0.0)));
        logger.log("");
    }

    public void compareClassSpecificFromMaps(Map<String, Map<String, Double>> j48ClassMetrics,
                                              Map<String, Map<String, Double>> rfClassMetrics) {
        logger.log("=== Class-specific Comparison ===");
        for (String className : j48ClassMetrics.keySet()) {
            logger.log("\nClass: " + className);

            Map<String, Double> j48Class = j48ClassMetrics.get(className);
            Map<String, Double> rfClass = rfClassMetrics.get(className);

            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "Precision",
                j48Class.getOrDefault("precision", 0.0),
                rfClass.getOrDefault("precision", 0.0)));
            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "Recall",
                j48Class.getOrDefault("recall", 0.0),
                rfClass.getOrDefault("recall", 0.0)));
            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "F-Measure",
                j48Class.getOrDefault("f-measure", 0.0),
                rfClass.getOrDefault("f-measure", 0.0)));
            logger.log(String.format("  %-20s | J48: %.4f | RF: %.4f", "ROC Area",
                j48Class.getOrDefault("roc", 0.0),
                rfClass.getOrDefault("roc", 0.0)));
        }
    }

    public void printConclusionFromMaps(Map<String, Double> j48Metrics, Map<String, Double> rfMetrics) {
        logger.log("\n=== CONCLUSION ===");
        double j48Accuracy = j48Metrics.getOrDefault("accuracy", 0.0);
        double rfAccuracy = rfMetrics.getOrDefault("accuracy", 0.0);

        String betterModel = rfAccuracy > j48Accuracy ? "Random Forest" : "J48 Decision Tree";
        double improvement = Math.abs(rfAccuracy - j48Accuracy);
        logger.log("Better performing model: " + betterModel);
        logger.log(String.format("Accuracy improvement: %.2f%%", improvement));
    }
}
