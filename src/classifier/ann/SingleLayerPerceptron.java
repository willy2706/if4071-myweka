package classifier.ann;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public abstract class SingleLayerPerceptron extends MyANN {
    private List<Double> input;
    private List<Double> weight;
    
    public SingleLayerPerceptron(int numInstance) {
        input = new ArrayList<Double>(numInstance);
        weight = new ArrayList<Double>(numInstance);
    };
    
    public List<Double> getInput() {
        return input;
    }

    public void setInput(List<Double> input) {
        this.input = input;
    }

    public List<Double> getWeight() {
        return weight;
    }

    public void setWeight(List<Double> weight) {
        this.weight = weight;
    }
}
