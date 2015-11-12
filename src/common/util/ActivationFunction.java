package common.util;

/**
 * Created by nim_13512065 on 11/12/15.
 */
public enum ActivationFunction {
    SIGMOID, SIGN, STEP;

    public double calculateOutput (double input) {
        if (this == ActivationFunction.SIGMOID) {

        } else if (this == ActivationFunction.SIGN){

        } else if (this == ActivationFunction.STEP) {

        }
        return 0.0;
    }
}
