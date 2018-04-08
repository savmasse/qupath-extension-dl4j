package qupath.lib.deep_learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.objects.PathObject;

import java.util.List;

/**
 * Simple QuPath wrapper for the ND4J DataSet. At this point this class will probably suffice,
 * but more child classes may be required in the future for different types of data (images, etc...)
 * 
 * The DataSet object is the standard object used in the  DL4J libary; so this class 
 * establishes a links between QuPath PathObjects and data used in Deep Learning.
 * 
 * @author Sam Vanmassenhove
 *
 */
class PathDataSet extends DataSet {
	protected static Logger logger  = LoggerFactory.getLogger(PathDataSet.class);

	private static final long serialVersionUID = -5858640838068356935L;
	
	protected PathObject pathObject;
	
	
	/**
	 * @param features The feature matrix
	 * @param labels Binarized array of matrices where the specified label has a one in the column.
	 */
	public PathDataSet (PathObject pathObject, INDArray features, INDArray labels) {
		
		super(features, labels);
		this.pathObject = pathObject;		

	}

	public PathDataSet (PathObject pathObject) {
		super();
		this.pathObject = pathObject;
	}

	public PathDataSet () {
		super();
	}
	
	public PathObject getPathObject () {
		return pathObject;
	}

	/**
	 * Get the features from the measurement list of the PathObject
	 */
	protected void convertFeatures (final List<String> measurements) {

		INDArray features;
		double [] d = new double[measurements.size()];
		int i = 0;

		for (String measurement : pathObject.getMeasurementList().getMeasurementNames()) {

			// If in the list, then add to the array
			if (measurements.contains(measurement)) {
				d[i] = pathObject.getMeasurementList().getMeasurementValue(measurement);
				i++;
			}
		}

		features = Nd4j.create(d);

		setFeatures(features);
	}

	protected void convertFeatures () {
		convertFeatures(pathObject.getMeasurementList().getMeasurementNames());
	}

}