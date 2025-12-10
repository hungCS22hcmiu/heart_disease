package com.example.heart_disease.classification.j48;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import com.example.heart_disease.classification.ClassifierBase;
import com.example.heart_disease.utils.Logger;

public class J48BalancedClassifier extends ClassifierBase {

    private boolean useBalancing;
    private Instances balancedData;

    public J48BalancedClassifier(Logger logger, boolean useBalancing) {
        super(logger, "J48 Decision Tree (Balanced)");
        this.useBalancing = useBalancing;
    }

    @Override
    public Classifier createClassifier() throws Exception {
        return new J48();
    }

    public Instances balanceData(Instances data) throws Exception {
        if (!useBalancing) {
            return data;
        }

        logger.log("\n=== Balancing Dataset (Resampling) ===");
        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0);
        resample.setNoReplacement(false);
        resample.setSampleSizePercent(100);
        resample.setInputFormat(data);

        Instances balancedData = Filter.useFilter(data, resample);

        logger.log("✓ Data Balanced!");
        logger.log("  - Original Instances: " + data.numInstances());
        logger.log("  - Balanced Instances: " + balancedData.numInstances());
        
        // Show class distribution
        logger.log("\n=== Class Distribution ===");
        int[] classCount = new int[data.numClasses()];
        for (int i = 0; i < balancedData.numInstances(); i++) {
            classCount[(int) balancedData.instance(i).classValue()]++;
        }
        for (int i = 0; i < data.numClasses(); i++) {
            double percentage = (classCount[i] / (double) balancedData.numInstances()) * 100;
            logger.log("  - Class " + i + " (" + data.classAttribute().value(i) + "): " + 
                      classCount[i] + " instances (" + String.format("%.2f%%", percentage) + ")");
        }
        logger.log("");

        return balancedData;
    }

    @Override
    public void buildModel(Instances data) throws Exception {
        balancedData = balanceData(data);
        super.buildModel(balancedData);
    }

    public Instances getBalancedData() {
        return balancedData;
    }
}
