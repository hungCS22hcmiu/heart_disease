package com.example.heart_disease.classification.randomforest;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import com.example.heart_disease.classification.ClassifierBase;
import com.example.heart_disease.utils.Logger;

public class RandomForestClassifier extends ClassifierBase {

    private boolean useBalancing;
    private Instances balancedData;

    public RandomForestClassifier(Logger logger, boolean useBalancing) {
        super(logger, "Random Forest");
        this.useBalancing = useBalancing;
    }

    @Override
    public Classifier createClassifier() throws Exception {
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        return rf;
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

