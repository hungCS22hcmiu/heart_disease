// src/main/java/com/example/heart_disease/DataAnalyzer.java
package com.example.heart_disease.step1;

import weka.core.Instances;
import weka.core.AttributeStats;
import weka.core.Attribute;

public class DataAnalyzer {
  
    /**
       Data analysis
     */
    public void performAnalysis(Instances data) {
        System.out.println("\n COMPREHENSIVE DATA ANALYSIS");
        printDatasetOverview(data);
        printAttributeDetails(data);
        printClassDistribution(data);
        printMissingValueReport(data);
        printCorrelationAnalysis(data);
    }

    private void printDatasetOverview(Instances data) {
        System.out.println("\n DATASET OVERVIEW ");
        System.out.println("Dataset Name: " + data.relationName());
        System.out.println("Total Instances: " + data.numInstances());
        System.out.println("Total Attributes: " + data.numAttributes());
        System.out.println("Class Attribute: " +
                (data.classIndex() >= 0 ? data.attribute(data.classIndex()).name() : "Not set"));
    }

    private void printAttributeDetails(Instances data) {
        System.out.println("\n ATTRIBUTE DETAILS ");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            AttributeStats stats = data.attributeStats(i);

            System.out.println((i+1) + ". " + attr.name() +
                    " [" + Attribute.typeToString(attr) + "]");

            if (attr.isNumeric()) {
                printNumericStats(data, i, stats);
            } else if (attr.isNominal()) {
                printNominalStats(data, i, stats, attr);
            }
            System.out.println("   Missing: " + stats.missingCount +
                    " (" + String.format("%.2f", (stats.missingCount * 100.0 / data.numInstances())) + "%)");
            System.out.println();
        }
    }

    private void printNumericStats(Instances data, int attrIndex, AttributeStats stats) {

        double min = stats.numericStats.min;
        double max = stats.numericStats.max;
        double mean = stats.numericStats.mean;
        double stdDev = stats.numericStats.stdDev;

        System.out.println("   Min: " + String.format("%.2f", min) +
                ", Max: " + String.format("%.2f", max) +
                ", Mean: " + String.format("%.2f", mean) +
                ", StdDev: " + String.format("%.2f", stdDev));
    }

    private void printNominalStats(Instances data, int attrIndex, AttributeStats stats, Attribute attr) {
        System.out.println("   Distinct values: " + attr.numValues());
        for (int j = 0; j < Math.min(attr.numValues(), 5); j++) { // Show top 5 values
            int count = stats.nominalCounts[j];
            System.out.println("     - " + attr.value(j) + ": " + count +
                    " (" + String.format("%.2f", (count * 100.0 / data.numInstances())) + "%)");
        }
        if (attr.numValues() > 5) {
            System.out.println("     ... and " + (attr.numValues() - 5) + " more values");
        }
    }

    private void printClassDistribution(Instances data) {
        if (data.classIndex() >= 0) {
            System.out.println("\n CLASS DISTRIBUTION ");
            Attribute classAttr = data.attribute(data.classIndex());
            AttributeStats classStats = data.attributeStats(data.classIndex());

            for (int i = 0; i < classAttr.numValues(); i++) {
                int count = classStats.nominalCounts[i];
                System.out.println(classAttr.value(i) + ": " + count +
                        " (" + String.format("%.2f", (count * 100.0 / data.numInstances())) + "%)");
            }
        }
    }

    private void printMissingValueReport(Instances data) {
        System.out.println("\n MISSING VALUE REPORT ");
        int totalMissing = 0;
        for (int i = 0; i < data.numAttributes(); i++) {
            AttributeStats stats = data.attributeStats(i);
            if (stats.missingCount > 0) {
                System.out.println(data.attribute(i).name() + ": " + stats.missingCount + " missing");
                totalMissing += stats.missingCount;
            }
        }
        System.out.println("Total missing values: " + totalMissing);
    }

    private void printCorrelationAnalysis(Instances data) {
        System.out.println("\n DATA QUALITY ASSESSMENT ---");
        System.out.println("Dataset completeness: " +
                String.format("%.2f", calculateCompleteness(data)) + "%");
        System.out.println("Recommended: Consider feature selection based on variance analysis");
    }

    private double calculateCompleteness(Instances data) {
        int totalValues = data.numInstances() * data.numAttributes();
        int missingValues = 0;

        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes(); j++) {
                if (data.instance(i).isMissing(j)) {
                    missingValues++;
                }
            }
        }

        return ((totalValues - missingValues) * 100.0) / totalValues;
    }
}