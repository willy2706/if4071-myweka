package classifier.ann;

import java.util.List;

/**
 * Created by nim_13512065 on 11/12/15.
 */
public class InputValue {
    private List<double[]> input;
    private List<Double> weight;

    public List<double[]> getInput() {
        return input;
    }

    public void addInput(List<double[]> value) {
        this.input.addAll(value);
    }

    public List<Double> getWeight() {
        return weight;
    }

    public void addWeight(List<Double> weight) {
        this.weight.addAll(weight);
    }
    
    
}
