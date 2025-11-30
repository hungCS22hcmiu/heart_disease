package com.example.heart_disease.classification;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import com.example.heart_disease.utils.Logger;

public abstract class ClassifierBase {

    protected Logger logger;
    protected Classifier classifier;
    protected String modelName;
    protected long buildTime;

    public ClassifierBase(Logger logger, String modelName) {
        this.logger = logger;
        this.modelName = modelName;
    }

    public abstract Classifier createClassifier() throws Exception;

    public Instances prepareData(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        logger.log("Class attribute: " + data.classAttribute().name());
        logger.log("Class attribute type: " + data.classAttribute().type());

        // 3. Convert numeric class to nominal
        logger.log("\n=== Converting Class Attribute to Nominal ===");
        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "last"; // Convert last attribute (class)
        convert.setOptions(options);
        convert.setInputFormat(data);
        data = Filter.useFilter(data, convert);

        return data;
    }

    public void buildModel(Instances data) throws Exception {
        logger.log("=== Building " + modelName + " ===");
        long startTime = System.currentTimeMillis();

        classifier = createClassifier();
        classifier.buildClassifier(data);

        buildTime = System.currentTimeMillis() - startTime;
        logger.log("✓ Model built successfully!");
        logger.log("✓ Build time: " + buildTime + " ms");
        logger.log("");
    }

    public void saveModel(String filepath) throws Exception {
        SerializationHelper.write(filepath, classifier);
        logger.log("✓ Model saved to: " + filepath);
    }

    public Classifier loadModel(String filepath) throws Exception {
        classifier = (Classifier) SerializationHelper.read(filepath);
        logger.log("✓ Model loaded from: " + filepath);
        return classifier;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public long getBuildTime() {
        return buildTime;
    }

    public String getModelName() {
        return modelName;
    }
}

