package qupath.lib.deep_learning;

import java.util.ArrayList;
import java.util.Collection;

import qupath.lib.objects.PathObject;

/**
 * Abstract parent class for the storing of high-level features.
 * 
 * @author Sam Vanmassenhove
 *
 */
public abstract class AbstractFeatureObject {
	
	private PathObject pathObject;
	private Collection <Double> featureList;
	
	public AbstractFeatureObject (final PathObject pathObject, final Collection <Double> featureList) {
		this.pathObject = pathObject;
		this.featureList = featureList;
	}
	
	public AbstractFeatureObject (final PathObject pathObject, final double [] features) {
		this.pathObject = pathObject;
		this.featureList = new ArrayList<Double>();
		
		for (int i = 0; i < features.length; i++) {
			featureList.add(features[i]);
		}
	}
	
	public Collection <Double> getFeatureList () {
		return this.featureList;
	}
	
	public PathObject getPathObject () {
		return this.pathObject;
	}
	
}
