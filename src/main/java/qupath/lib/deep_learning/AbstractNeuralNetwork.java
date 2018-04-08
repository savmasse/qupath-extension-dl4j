package qupath.lib.deep_learning;

import java.io.File;
import java.io.IOException;
import java.util.*;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.gui.QuPathGUI;

/**
 * Abstract parent class for all neural network implementations using DL4J. This class should
 * wrap a simple DL4J MultiLayerNetwork structure so these networks can be used more easily in 
 * the Qupath environment.
 *
 * All shared neural network functionality is handled here.
 * 
 * @author Sam Vanmassenhove
 *
 */
public abstract class AbstractNeuralNetwork <T extends PathDataSet> {
	
	protected static final Logger logger = LoggerFactory.getLogger(AbstractNeuralNetwork.class);
	protected static final int seed = 123; // Random number generator seed so results can be reproduced

	// The appendix 'Set' is used here because it's a common term in the literature;
	// not because the data structure is a collections.Set
	protected List <T> trainingSet;
	protected List <T> testSet;

	protected MultiLayerConfiguration conf;
	protected MultiLayerNetwork model;
	protected Evaluation eval;

	protected DataSetIterator trainingSetIter;
	protected DataSetIterator testSetIter;

	protected DataNormalization normalization;

	protected enum NormalizationMethod {
	    MinMax,
        Standardize
    }
	
	/**
	 * Constructor loads a list of samples to feed into the network.
	 * @param sampleList List of all samples to feed the network
	 */
	public AbstractNeuralNetwork (final List <T> sampleList) {

		// Put all data in the training set
		this.trainingSet = sampleList;
		this.trainingSetIter = new ListDataSetIterator<T>(sampleList);
	}

	/**
	 * Split the data into a training set and a set for testing. The testing set will
	 * not be used to train the model and will act as unseen data for the evaluation.
	 *
	 * @param trainPercentage Percentage of the data to use for training, the rest will
	 *                        be used for testing.
	 */
	public void splitTrainTest (final double trainPercentage) {

		// Shuffle the training list randomly
		Collections.shuffle(trainingSet, new Random(seed));

		// Now split the list in two
		testSet = testSet.subList((int) trainPercentage*trainingSet.size(), trainingSet.size()-1);
		trainingSet = trainingSet.subList(0, (int) trainPercentage*trainingSet.size());

		// Create iterators
		testSetIter = new ListDataSetIterator<T>(testSet);
		trainingSetIter = new ListDataSetIterator<T>(trainingSet);

		// Normalize if a normalization method was set
		if (normalization == null) {
			testSetIter.setPreProcessor(normalization);
			trainingSetIter.setPreProcessor(normalization);
		}
	}
	
	/**
	 * Create the model and add the listeners
	 */
	public void createModel (final int printScoreIterations) {
		
		if (conf == null) {
			logger.info("Configuration was not set. Could not create the model...");
			return;
		}

		// Create the actual model
		model = new MultiLayerNetwork(conf);
		model.init();
		ScoreIterationListener scoreListener = new ScoreIterationListener(printScoreIterations);
		model.setListeners(Arrays.asList((IterationListener) scoreListener));

		logger.info("Created neural network model.");
	}

	/**
	 * Set and apply the normalization algorithm to be used in the model. Normalization must be called
	 * before splitting the data set into training and testing data.
	 *
	 * @param method The normalization method used
	 */
	public void normalize (NormalizationMethod method) {

		logger.info("Normalizing data...");

	    if (method.equals(NormalizationMethod.MinMax)) {
            normalization = new NormalizerMinMaxScaler();
        }
        else if (method.equals(NormalizationMethod.Standardize)) {
	        normalization = new NormalizerStandardize();
        }

		// Fit the normalizer to the given data
		normalization.fit(trainingSetIter);
	}

	/**
	 * Fit the model to the training data.
	 * 
	 * @param epochCount Amount of times the network will train on the whole dataset.
	 */
	public void fitModel (int epochCount) {
		
		if (model == null) {
			logger.info("Model was not initialized - cannot start training.");
			return;
		}
		if (trainingSetIter == null) {
			logger.info("Iterator was not initialized - cannot start training.");
			return;
		}
		if (epochCount <= 0) {
			epochCount = 1;
		}
		
		logger.info("Training model...");
		
		for (int i = 0; i < epochCount; i++) {
			model.fit(trainingSetIter);
		}
		
		logger.info("Model was trained");
	}
	
	/**
	 * Fit model to the training data.
	 */
	public void fitModel () {
		fitModel(1);
	}
	
	/**
	 * For now the models will be hardcoded in each child class as the deep learning is currently only used for a single purpose.
	 * 
	 * TODO Consider adding editor so user can create their own networks in the QuPath interface...
	 * 
	 * @param iterations Amount of iterations for the network training.
	 */
	public abstract void buildConfiguration(final int iterations);
		
	/**
	 * Evaluate the model by applying it to the test set and analyzing the results. Will print
	 * the full analysis in the log.
	 */
	public void evaluateModel () {
		
		// If the test set remains uninitialized, we know that there's no testing required
		if (testSetIter == null || testSet == null) {
			logger.info("This network was not tested. Either this was deliberate, or something went wrong...");
			return;
		}		
		// Create the evaluator
		eval = model.evaluate(testSetIter);
	
		logger.info("======== Neural network evaluated ======= \n");
	}
	
	/**
	 * Use DL4J functionality to store the trained model on the disk for later use. Should be stored as a zip file.
	 *
	 * @throws IOException 
	 */
	public void saveModel () throws IOException {

		if (model == null) {
			logger.info("Model was not instantiated - could not be saved.");
			return;
		}
		
		// Get a file to save to: qupath.getDialogHelper
		File saveFile = QuPathGUI.getInstance().getDialogHelper().promptToSaveFile("Save the neural network", new File("..\\"), "NeuralNetwork", null , "zip");

		if (saveFile == null) {
			logger.info("No location selected - file was not saved !");
			return;
		}
		
		ModelSerializer.writeModel(model, saveFile, true); // Always save the updater (this is the extra boolean) so model can be trained again once reloaded.
	}
	
	/**
	 * Load a trained model from a previous project from the disk. Will always be a zip file.
	 * 
	 * @throws IOException
	 */
	public void loadModel () throws IOException {
		
		// Open a filechooser to select the model.
		File loadFile = QuPathGUI.getInstance().getDialogHelper().promptForFile("Load neural network", null, "zip");
		
		if (loadFile == null) {
			logger.info("No file selected - nothing was loaded !");
			return;
		}

		// Set the current model to this newly loaded model
		model = ModelSerializer.restoreMultiLayerNetwork(loadFile);
	}
	
	/**
	 * Get the output from the model for a specified sample
	 * @param sample
	 */
	public INDArray output (T sample) {
		return model.output(sample.getFeatureMatrix());
	}

	/**
	 * Get output from the model for a list of samples
	 * @param sampleList
	 */
	public INDArray output (List <T> sampleList) {
		DataSetIterator iter = new ListDataSetIterator<T>(sampleList);
		return model.output(iter);
	}
	
	/**
	 * Get the features from the last layer of the neural network for a specified sample 
	 */
	public INDArray activateLastLayer (T sample) {
		 return model.getOutputLayer().activate(sample.getFeatureMatrix());
	}

	/**
	 * Feeds the input into the network and gets the activation for each layer up to the
	 * given layer. The data is assumed to not yet be normalized.
	 *
	 * @param layer Last layer to activate.
	 * @param sample The dataset used as input.
	 * @return List of the activations of the layer up to the requested layer.
	 */
	public List<INDArray> feedForwardActivation (final int layer, final T sample) {
		INDArray normalized = sample.getFeatureMatrix();
		normalization.transform(normalized);
		return model.feedForwardToLayer(layer, normalized);
	}

	/**
	 * Return the name of this type of network.
	 * @return @{@link String} name
	 */
	public abstract String getName ();

	/**
	 * Return a short description of this type of network
	 * @return @{@link String} description
	 */
	public abstract String getDescription ();

}
