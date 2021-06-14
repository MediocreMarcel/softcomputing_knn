package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.*;

public class HiddenLayerNeuron extends Neuron{

    public HiddenLayerNeuron(Map<Neuron, Double> parents) {
        this.parents = new HashMap<>(parents);
    }

    public HiddenLayerNeuron(Map<Neuron, Double> parents, boolean isBias) {
        this.parents = parents;
        this.isBias = isBias;
    }

    public static Map<Neuron, Double> initHiddenLayer(Map<Neuron, Double> parents, int[] structureNN){

        List<Neuron> currentLayer = new LinkedList<>();
        currentLayer.add(new HiddenLayerNeuron(parents, true));
        for (int i = 1; i < structureNN[0]; i++) {
            currentLayer.add(new HiddenLayerNeuron(parents));
        }

        Map<Neuron, Double> currentLayerMap = createLayerMap(currentLayer);

        if (structureNN.length > 1){
            Map<Neuron, Double> childrenMap = initHiddenLayer(currentLayerMap, Arrays.copyOfRange(structureNN, 1, structureNN.length));

            //set Child for each neuron in this starting layer
            for (Neuron neuron : currentLayer) {
                neuron.children = new HashMap<>(childrenMap);//create a unique map for each neuron
            }
        } else if (structureNN.length == 1){
            EndingLayerNeuron endingLayerNeuron = new EndingLayerNeuron(createLayerMap(currentLayer));
            Map<Neuron, Double> endingLayerMap = new HashMap<>();
            endingLayerMap.put(endingLayerNeuron, 0.0);

            for (Neuron neuron:currentLayer) {
                neuron.children = endingLayerMap;
            }
        }


        return currentLayerMap;
    }
}
