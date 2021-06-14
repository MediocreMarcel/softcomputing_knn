package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.*;

public class Neuron {

    protected Map<Neuron, Double> parents;
    protected Map<Neuron, Double> children;
    protected boolean isBias = false;
    protected double input;
    protected double output;
    protected double delta;

    protected static Map<Neuron, Double> createLayerMap(List<Neuron> layer) {
        Map<Neuron, Double> layerMap = new HashMap<>();
        for (Neuron neuron : layer) {
            if (neuron.isBias){
                layerMap.put(neuron, 1.0);
            }
            layerMap.put(neuron, 0.0);
        }
        return layerMap;
    }

    public static double getRandomWeight(){
        double weight = Math.random();
        return Math.random() < 0.5 ? -weight: weight;
    }


    protected void setParentWeight(Neuron neuron, double weight){
            if(this.parents.containsKey(neuron)){
                this.parents.put(neuron, weight);
            } else if(neuron.isBias){
                return; //Not sure if this is correct
            } else{
                System.exit(1);
            }
        }

    protected void setParentsWeight(Map<Neuron, Double> map){
        for (Neuron neuron: map.keySet()){
            if(this.parents.containsKey(neuron)){
                this.parents.put(neuron, map.get(neuron));
            } else{
                System.exit(1);
            }
        }
    }

    protected void setChildrensWeight(Map<Neuron, Double> map){
        for (Neuron neuron: map.keySet()){
            if(this.children.containsKey(neuron)){
                this.children.put(neuron, map.get(neuron));
            } else{
                System.exit(1);
            }
        }
    }

    public static void initWeights(List<Neuron> neuronLayer) {
        for (Neuron neuron : neuronLayer) {
            if (neuron instanceof EndingLayerNeuron){
                return;
            }
            neuron.children.keySet().stream().forEach(x -> {
                double randomWeight = neuron.getRandomWeight();
                neuron.children.put(x, randomWeight);
                x.setParentWeight(neuron, randomWeight);
                initWeights(x);
            });
        }
    }

    public static void initWeights(Neuron currentNeuron) {
        if (currentNeuron instanceof EndingLayerNeuron){
            return;
        }

        Map<Neuron, Double> neuronChildren = currentNeuron.children;
        List<Neuron> childNeurons = new ArrayList<>(neuronChildren.keySet());
        for (Neuron childNeuron: childNeurons){
            double randomWeight = getRandomWeight();
            neuronChildren.put(childNeuron, randomWeight);
            childNeuron.setParentWeight(currentNeuron, randomWeight);
            initWeights(childNeurons);
        }
    }
}
