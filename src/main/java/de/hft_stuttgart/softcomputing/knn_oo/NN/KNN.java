package de.hft_stuttgart.softcomputing.knn_oo.NN;

import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.Neuron;
import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.StartingLayerNeuron;

import java.util.List;


public class KNN {
    public enum NetworkState {TRAIN, EVALUATE};

    public static double[] currentDataRow;
    public static NetworkState networkState = NetworkState.TRAIN;
    public static double currentOutcome = 0.0;

    List<Neuron> startingNeurons = null;

    public KNN(int inputCount, int[] numberOfNeuronsPerHiddenLayer) {
        startingNeurons = StartingLayerNeuron.initStartingLayer(inputCount, numberOfNeuronsPerHiddenLayer);
    }

    public void train(double[][] trainingData, int maxEpoch) {
        boolean stop = false;
        int epoch = 0;

        StartingLayerNeuron.initWeights(startingNeurons);

        while (!stop){
            epoch++;

            for (int i = 0; i < trainingData.length; i++) {
                currentDataRow = trainingData[i];
                for (int j = 0; j < trainingData[i].length - 1 /*ignore outcome*/; j++)
                    startingNeurons.get(j+1).setOutput(currentDataRow[j]); //+1 because bias is at position 0

                startingNeurons.get(0).startForwardOnChildren();
            }

            System.out.println("Epoche: " + epoch);

            if (epoch == maxEpoch)
                stop = true;
        }


        networkState = NetworkState.EVALUATE;
    }

    public void evaluate(double[][] data) {
        int falsePositive = 0;
        int truePositive = 0;
        int falseNegative = 0;
        int trueNegative = 0;

        int amountPositives = 0;
        int amountNegative = 0;


        for (int i = 0; i < data.length; i++) {
            currentDataRow = data[i];
            for (int j = 0; j < data[i].length - 1 /*ignore outcome*/; j++)
                startingNeurons.get(j+1).setOutput(currentDataRow[j]);
            startingNeurons.get(0).startForwardOnChildren();

            double expectedOutcome = currentDataRow[currentDataRow.length-1];

            if (currentOutcome >= 0.5 && expectedOutcome == 1.0){
                truePositive++;
                amountPositives++;
            } else if(currentOutcome >= 0.5 && expectedOutcome == 0.0) {
                falsePositive++;
                amountNegative++;
            } else if(currentOutcome < 0.5 && expectedOutcome == 1.0){
                falseNegative++;
                amountPositives++;
            } else if (currentOutcome < 0.5 && expectedOutcome == 0.0) {
                trueNegative++;
                amountNegative++;
            } else {
                System.exit(2);
            }
        }

        System.out.println("Anzahl Muster:  \t" + data.length);
        System.out.println("Anzahl Positiv: \t" + amountPositives);
        System.out.println("Anzahl Negativ: \t" + amountNegative);
        System.out.println("Anteil Positiv: \t" + (double)amountPositives/(double)data.length);
        System.out.println("Anteil Negativ: \t" + (double)amountNegative/(double)data.length);

        System.out.println("Genauigkeit  :  \t" + (double)(truePositive+trueNegative)/(double)data.length);
        System.out.println("Trefferquote:   \t" + (double)(truePositive)/(double)(truePositive+falseNegative));
        System.out.println("Ausfallrate :   \t" + (double)(falsePositive)/(double)(trueNegative+falsePositive));

        System.out.println("richtigPositiv: \t" + truePositive);
        System.out.println("falsch Negativ: \t" + falseNegative);
        System.out.println("richtigNegativ: \t" + trueNegative);
        System.out.println("falsch Positiv: \t" + falsePositive);
    }
}
