package qupath.lib.deep_learning;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkExample {

    private static final Logger logger = LoggerFactory.getLogger(NeuralNetworkExample.class);

    public NeuralNetworkExample () {}

    public static void run () throws IOException {

        System.out.println("Started neural network test...");

        // Create an sample list of data
        PathDataSet pds = new PathDataSet();
        List<PathDataSet> dataSetList = new ArrayList<>();
        dataSetList.add(pds);

        AutoEncoderNetwork an = new AutoEncoderNetwork(dataSetList, 1000, 1000);

        // Build configuration
        an.buildConfiguration(100);

        // Create the model
        an.createModel(10);

        // Save the model
        an.saveModel();

        logger.info("Succesfully finished neural network test");
    }
}
