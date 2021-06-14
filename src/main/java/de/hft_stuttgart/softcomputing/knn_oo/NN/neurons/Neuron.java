package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Neuron {

    public Map<Neuron, Double> parents;
    public Map<Neuron, Double> children;
    public boolean isBias = false;
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

    public double getRandomWeight(){
        double weight = Math.random();
        return Math.random() < 0.5 ? -weight: weight;
    }


    protected void setParentWeight(Neuron neuron, double weight){
            if(this.parents.containsKey(neuron)){
                this.parents.put(neuron, weight);
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
}
