package de.hft_stuttgart.softcomputing.knn_oo.NN;

import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.Neuron;
import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.StartingLayerNeuron;

import java.util.List;

/**
 * Class handles the starting layer, the training and the evaluation
 */
public class KNN {
    public enum NetworkState {TRAIN, EVALUATE};

    /**
     * Holds the current data row out of the dataset
     */
    public static double[] currentDataRow;

    /**
     *  state of the network training or evluation
     */
    public static NetworkState networkState = NetworkState.TRAIN;

    /**
     * holds the value of the ending layer neuron
     */
    public static double currentOutcome = 0.0;

    /**
     * holds starting neurons
     */
    List<Neuron> startingNeurons = null;

    /**
     * initializes the network
     * @param inputCount number of dataset parameters => Number of starting layer neurons
     * @param numberOfNeuronsPerHiddenLayer array holding the number of neurons per hidden layer (incl. Bias)
     */
    public KNN(int inputCount, int[] numberOfNeuronsPerHiddenLayer) {
        startingNeurons = StartingLayerNeuron.initStartingLayer(inputCount, numberOfNeuronsPerHiddenLayer);
    }

    /**
     * starts the training of the Network
     * @param trainingData data that the network should train from
     * @param maxEpoch number of epochs that the network should train
     */
    public void train(double[][] trainingData, int maxEpoch) {
        boolean stop = false;
        int epoch = 0;

        //init weights with random values
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

        //set state to evaluate after the network finished training
        networkState = NetworkState.EVALUATE;
    }

    /**
     * evaluates the network by comparing the outcome of the network against the expected result
     * @param data data that should be used to perform the evaluation
     */
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
