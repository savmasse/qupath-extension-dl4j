package qupath.lib.deep_learning;

import java.util.Collections;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handles a simple autoencoder model to used in feature extraction
 * 
 * @author Sam Vanmassenhove
 *
 */
public class AutoEncoderNetwork <T extends PathDataSet> extends AbstractNeuralNetwork <T> {
	
	private static final Logger logger = LoggerFactory.getLogger(AutoEncoder.class);
	private int inputSize, outputSize;

	/**
	 * TODO Let the constructor discover the input and output size from the data !
	 * @param sampleList
	 * @param inputSize
	 * @param outputSize
	 */
	public AutoEncoderNetwork (final List <T> sampleList, final int inputSize, final int outputSize) {
		super(sampleList);
		this.inputSize = inputSize;
		this.outputSize = outputSize;
	}

	/**
	 * This is a standard configuration from an autoencoder example in the DL4J documentation...
	 */
	@Override
	public void buildConfiguration(final int iterations) {

		conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .iterations(iterations)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new AutoEncoder.Builder().nIn(inputSize).nOut(inputSize/2)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .corruptionLevel(0.3)
                        .build())
                //.layer(1, new AutoEncoder.Builder().nIn(500).nOut(250)
                       // .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                       // .corruptionLevel(0.3)

                       // .build())
               // .layer(2, new AutoEncoder.Builder().nIn(250).nOut(200)
                 //       .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                   //     .corruptionLevel(0.3)
                     //   .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(inputSize/2).nOut(outputSize).build())
                .pretrain(true).backprop(false)
                .build();
	
		logger.info ("Built autoencoder configuration.");
	}

	public String getName () {
		return "Autoencoder";
	}

	public String getDescription () {
		return "Simple autoencoder example";
	}
}
