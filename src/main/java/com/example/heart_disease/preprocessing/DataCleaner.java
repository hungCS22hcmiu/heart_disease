package com.example.heart_disease.preprocessing;

import weka.core.Instances;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class DataCleaner {

    /**
     * Complete data cleaning pipeline for heart disease dataset.
     */
    public Instances cleanData(Instances data) throws Exception {
        System.out.println("\n  DATA CLEANING PIPELINE");

        Instances cleanedData = new Instances(data);

        // FIX 1: Correct column names
        cleanedData = renameAttributes(cleanedData);

        // Step 1: Handle missing values (Imputation)
        cleanedData = handleMissingValues(cleanedData);

        // Step 2: Categorical Encoding
        cleanedData = encodeNominalAttributes(cleanedData);

        // Step 3: Remove constant attributes (Corrected method)
        cleanedData = removeConstantAttributes(cleanedData);

        // Step 4: Remove duplicate instances
        cleanedData = removeDuplicates(cleanedData);

        System.out.println(" Data cleaning completed successfully.");
        System.out.println("Final dataset: " + cleanedData.numInstances() +
                " instances, " + cleanedData.numAttributes() + " attributes (Now fully numerical)");

        return cleanedData;
    }

    /**
     Remove single quotes from attribute names.
     */
    private Instances renameAttributes(Instances data) {
        System.out.println(" Fixing attribute names (removing quotes)...");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            String name = attr.name();

            if (name.startsWith("'") && name.endsWith("'")) {
                String newName = name.substring(1, name.length() - 1);
                data.renameAttribute(attr, newName);
            }
        }
        System.out.println(" Attribute names cleaned.");
        return data;
    }

    /**
     Convert all nominal (categorical) attributes to binary features.
     */
    private Instances encodeNominalAttributes(Instances data) throws Exception {
        System.out.println(" Encoding nominal attributes to binary features...");

        NominalToBinary nominalToBinaryFilter = new NominalToBinary();
        nominalToBinaryFilter.setInputFormat(data);

        Instances encodedData = Filter.useFilter(data, nominalToBinaryFilter);

        System.out.println(" Attributes before encoding: " + data.numAttributes());
        System.out.println(" Attributes after encoding: " + encodedData.numAttributes());

        return encodedData;
    }

    /**
     Remove attributes with zero variance (constant values).
     */
    private Instances removeConstantAttributes(Instances data) throws Exception {
        StringBuilder constantAttrs = new StringBuilder();

        for (int i = 0; i < data.numAttributes(); i++) {
            // FIX: Removed the invalid .isBinary() check.
            // All numerical attributes, including the new binary ones, are covered by .isNumeric().
            if (data.attribute(i).isNumeric()) {
                if (data.variance(i) == 0) {
                    if (constantAttrs.length() > 0) constantAttrs.append(",");
                    constantAttrs.append(i + 1);
                    System.out.println(" Removing constant attribute: " + data.attribute(i).name());
                }
            }
        }

        if (constantAttrs.length() > 0) {
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices(constantAttrs.toString());
            removeFilter.setInputFormat(data);
            return Filter.useFilter(data, removeFilter);
        }

        return data;
    }

    /**
     Handle missing values using Weka's ReplaceMissingValues filter (Imputation).
     */
    private Instances handleMissingValues(Instances data) throws Exception {
        int missingBefore = countMissingValues(data);

        if (missingBefore > 0) {
            System.out.println(" Handling " + missingBefore + " missing values...");

            ReplaceMissingValues replaceFilter = new ReplaceMissingValues();
            replaceFilter.setInputFormat(data);
            Instances cleanedData = Filter.useFilter(data, replaceFilter);

            int missingAfter = countMissingValues(cleanedData);
            System.out.println(" Missing values handled: " + missingBefore + " → " + missingAfter);

            return cleanedData;
        }

        System.out.println(" No missing values found. Imputation skipped.");
        return data;
    }

    /**
     Remove duplicate instances.
     */
    private Instances removeDuplicates(Instances data) {
        java.util.HashSet<String> uniqueInstances = new java.util.HashSet<>();
        Instances cleanedData = new Instances(data, 0);
        int duplicatesRemoved = 0;

        for (int i = 0; i < data.numInstances(); i++) {
            String instanceString = data.instance(i).toString();
            if (!uniqueInstances.contains(instanceString)) {
                uniqueInstances.add(instanceString);
                cleanedData.add(data.instance(i));
            } else {
                duplicatesRemoved++;
            }
        }

        if (duplicatesRemoved > 0) {
            System.out.println(" Removed " + duplicatesRemoved + " duplicate instances");
        }

        return cleanedData;
    }

    /**
     Count total missing values in dataset.
     */
    private int countMissingValues(Instances data) {
        int missingCount = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes(); j++) {
                if (data.instance(i).isMissing(j)) {
                    missingCount++;
                }
            }
        }
        return missingCount;
    }
}

