package com.example.heart_disease.utils;

import java.io.PrintWriter;

public class Logger {

    private PrintWriter writer;
    private boolean consoleOutput;

    public Logger(PrintWriter writer, boolean consoleOutput) {
        this.writer = writer;
        this.consoleOutput = consoleOutput;
    }

    public Logger(PrintWriter writer) {
        this(writer, true);
    }

    public void log(String message) {
        if (consoleOutput) {
            System.out.println(message);
        }
        if (writer != null) {
            writer.println(message);
            writer.flush();
        }
    }

    public void close() {
        if (writer != null) {
            writer.close();
        }
    }
}

