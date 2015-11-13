package classifier.ann;

/**
 * Created by nim_13512065 on 11/12/15.
 */
public class InputValue {
    private double value;
    private double weight;

    public InputValue() {

    }

    public InputValue (double value, double weight) {
        setValue(value);
        setWeight(weight);
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
