package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import de.hft_stuttgart.softcomputing.knn_oo.NN.KNN;

import java.util.Map;

public class EndingLayerNeuron extends Neuron {

    public EndingLayerNeuron(Map<Neuron, Double> parents) {
        this.parents = parents;
    }

    @Override
    protected void calculateForward() {
        input = 0.0;
        parents.forEach((neuron,weight) -> {
            if (neuron.isBias){
                return;
            }
            input += neuron.output*weight;
        });
        output = sigmoidFunction(input);

        KNN.currentOutcome = output;

        if (KNN.networkState.equals(KNN.NetworkState.TRAIN)){
            calculateBackward();
        }
    }

    @Override
    protected void calculateBackward() {
        delta = derivativeSigmoidFunction(input) * (output-KNN.currentDataRow[KNN.currentDataRow.length-1]);
        startBackwardOnParents();
    }


}
