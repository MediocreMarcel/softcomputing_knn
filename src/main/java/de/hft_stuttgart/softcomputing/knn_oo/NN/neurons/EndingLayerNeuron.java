package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.Map;

public class EndingLayerNeuron extends Neuron {

    public EndingLayerNeuron(Map<Neuron, Double> parents) {
        this.parents = parents;
    }
}
