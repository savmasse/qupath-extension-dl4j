package qupath.lib.deep_learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class PCAExample {

    public static void run () {

        //Create points as NDArray instances
        List<INDArray> ndArrays = Arrays.asList(
                Nd4j.create(new float [] {-1.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F}),
                Nd4j.create(new float [] {-1.0F, 1.0F, -1.0F, -3.0F, 2.0F, -1.0F}),
                Nd4j.create(new float [] {1.0F, 1.0F, -1.0F, -1.0F, -1.0F, 1.0F}));

        //Create matrix of points (rows are observations; columns are features)
        INDArray matrix = Nd4j.create(ndArrays, new int [] {3,6});
        System.out.println("" + matrix.toString());

        //Execute PCA - again to 2 dimensions
        INDArray factors = PCA.pca(matrix, 4, false);

        System.out.println(" ========== ");
        System.out.println("Finished: " + factors.toString());

        // Convert to java array
        double [] d = factors.dup().transpose().data().asDouble();
    }
}
