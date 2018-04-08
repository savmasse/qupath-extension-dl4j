
package qupath.lib.deep_learning;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_ml;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TSNEExample  {
    public static void main(String [] args) throws Exception {
        int initial_dims = -1;
        double perplexity = 20.0;
        int iterations = 10;
        double theta = 0.500000;

        // Some random data
        String [] labels = new String[5000];
        double [] [] X = new double[5000][100];
        for (int i = 0; i < 5000; i++) {
            for (int j = 0; j < 100; j++) {
                //X[i][j] = Math.random();
                X[i][j] = Math.random();
            }
            labels[i] = new String ("" + i);
        }

        /*
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(iterations)
                .perplexity(perplexity)
                .theta(theta)
                .normalize(true)
                .build();
        */
        Tsne tsne = new Tsne.Builder()
                .setMaxIter(iterations)
                .perplexity(perplexity)
                .normalize(false)
                .build();

        //tsne.plot(Nd4j.create(X), 2, Arrays.asList(labels), "C:\\Users\\SamVa\\Desktop\\tsne.csv");
        //tsne.fit(Nd4j.create(X));
        //INDArray result = tsne.getData();
        INDArray result = tsne.calculate(Nd4j.create(X), 2, 20);

        //System.out.println(result.shape()[0] + ", " + result.shape()[1]);
        System.out.println(result.toString());

        //tsne.fit(Nd4j.create(X));
        //tsne.saveAsFile(Arrays.asList(labels), "C:\\Users\\SamVa\\Desktop\\tsne.csv");
    }
}