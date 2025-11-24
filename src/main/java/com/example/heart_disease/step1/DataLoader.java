package com.example.heart_disease.step1;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver; // NEW Import
import java.io.File;

public class DataLoader {

    /**
      Load dataset from CSV file
     */
    public Instances loadCSV(String filePath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        Instances data = loader.getDataSet();
        System.out.println("✓ Dataset loaded: " + data.numInstances() + " instances, " +
                data.numAttributes() + " attributes");
        return data;
    }

    /**
      Save dataset to CSV format
     */
    public void saveAsCSV(Instances data, String outputPath) throws Exception { // RENAMED method
        CSVSaver saver = new CSVSaver(); // Changed from ArffSaver
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
        System.out.println("✓ Dataset saved as CSV: " + outputPath);
    }
    /**
     Save dataset to ARFF format (NEW METHOD)
     */
    public void saveAsARFF(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
        System.out.println("✓ Dataset saved as ARFF: " + outputPath);
    }

    /**
      Split data into training and testing sets (80-20 split)
     */
    public Instances[] splitTrainTest(Instances data, double trainRatio) throws Exception {
        data.randomize(new java.util.Random(42));
        int trainSize = (int) Math.round(data.numInstances() * trainRatio);
        int testSize = data.numInstances() - trainSize;

        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        System.out.println("✓ Data split - Training: " + train.numInstances() +
                ", Testing: " + test.numInstances());

        return new Instances[]{train, test};
    }
}