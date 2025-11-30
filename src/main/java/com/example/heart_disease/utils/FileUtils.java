package com.example.heart_disease.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class FileUtils {

    public static String resolveDataPath(String path) {
        File file = new File(path);
        if (file.exists()) {
            return path;
        }

        String resourcePath = "src/main/resources/" + new File(path).getName();
        file = new File(resourcePath);
        if (file.exists()) {
            return resourcePath;
        }

        return path;
    }

    public static PrintWriter createOutputWriter(String outputPath) throws Exception {
        File outputFile = new File(outputPath);
        outputFile.getParentFile().mkdirs();
        return new PrintWriter(new FileWriter(outputFile));
    }

    public static String getOutputPath(String filename) {
        return "docs/output/" + filename;
    }

    public static String getResourcePath(String filename) {
        return "src/main/resources/" + filename;
    }
}

