package classifier.ann;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public abstract class SingleLayerPerceptron extends MyANN {
    private List<InputValue> weight;
    
    public SingleLayerPerceptron(int numInstance) {
        weight = new ArrayList<InputValue>(numInstance);
    };
}
