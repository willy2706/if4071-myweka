package common.util;

/**
 * Created by nim_13512065 on 11/12/15.
 */
public enum ActivationFunction {
    SIGMOID, SIGN, STEP;

    public double calculateOutput (double input) {
        if (this == ActivationFunction.SIGMOID) {
            double penyebut = 1.0 + Math.exp(-input);
            double pembilang = 1.0;
            return pembilang/penyebut;
        } else if (this == ActivationFunction.SIGN){
            
        } else if (this == ActivationFunction.STEP) {

        }
        return 0.0;
    }
}