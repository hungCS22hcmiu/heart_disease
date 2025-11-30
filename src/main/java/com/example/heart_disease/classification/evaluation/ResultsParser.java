package com.example.heart_disease.classification.evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

public class ResultsParser {

    public static Map<String, Double> parseMetrics(String filePath) {
        Map<String, Double> metrics = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.contains("Accuracy:") && line.contains("%")) {
                    String value = line.split(":")[1].trim().replace("%", "");
                    metrics.put("accuracy", Double.parseDouble(value));
                }
                else if (line.contains("Kappa Statistic:")) {
                    String value = line.split(":")[1].trim();
                    metrics.put("kappa", Double.parseDouble(value));
                }
                else if (line.contains("Mean Absolute Error:")) {
                    String value = line.split(":")[1].trim();
                    metrics.put("mae", Double.parseDouble(value));
                }
                else if (line.contains("Root Mean Squared Error:")) {
                    String value = line.split(":")[1].trim();
                    metrics.put("rmse", Double.parseDouble(value));
                }
                else if (line.contains("Build time:") || line.contains("Cross-validation time:")) {
                    String value = line.split(":")[1].trim().replace("ms", "").trim();
                    if (line.contains("Cross-validation")) {
                        metrics.put("cv_time", Double.parseDouble(value));
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error parsing file: " + filePath);
            e.printStackTrace();
        }

        return metrics;
    }

    public static Map<String, Map<String, Double>> parseClassMetrics(String filePath) {
        Map<String, Map<String, Double>> classMetrics = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            String currentClass = null;

            while ((line = reader.readLine()) != null) {
                if (line.startsWith("Class:")) {
                    currentClass = line.split(":")[1].trim();
                    classMetrics.put(currentClass, new HashMap<>());
                }
                else if (currentClass != null) {
                    if (line.contains("Precision:")) {
                        String value = line.split(":")[1].trim();
                        classMetrics.get(currentClass).put("precision", Double.parseDouble(value));
                    }
                    else if (line.contains("Recall:")) {
                        String value = line.split(":")[1].trim();
                        classMetrics.get(currentClass).put("recall", Double.parseDouble(value));
                    }
                    else if (line.contains("F-Measure:")) {
                        String value = line.split(":")[1].trim();
                        classMetrics.get(currentClass).put("f-measure", Double.parseDouble(value));
                    }
                    else if (line.contains("ROC Area:")) {
                        String value = line.split(":")[1].trim();
                        classMetrics.get(currentClass).put("roc", Double.parseDouble(value));
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error parsing class metrics from: " + filePath);
            e.printStackTrace();
        }

        return classMetrics;
    }
}

